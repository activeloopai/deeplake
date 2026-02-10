// Include libintl.h first to avoid conflicts with PostgreSQL's gettext macro
#include <libintl.h>

// DuckDB headers must come before PostgreSQL headers to avoid namespace pollution
#include <duckdb.hpp>

#include "duckdb_executor.hpp"
#include "duckdb_pg_convert.hpp"
#include "is_pure_count_star_query.hpp"

#include <deeplake_api/dataset_view.hpp>

#include "deeplake_executor.hpp"
#include "pg_deeplake.hpp"
#include "pg_version_compat.h"
#include "reporter.hpp"
#include "table_storage.hpp"

extern "C" {
#include <postgres.h>
#include <access/table.h>
#include <catalog/pg_type_d.h>
#include <executor/executor.h>
#include <executor/tuptable.h>
#include <nodes/extensible.h>
#include <nodes/makefuncs.h>
#include <nodes/nodeFuncs.h>
#include <nodes/plannodes.h>
#include <optimizer/optimizer.h>
#include <optimizer/planner.h>
#include <parser/parsetree.h>
#include <utils/builtins.h>
#include <utils/lsyscache.h>
#include <utils/memutils.h>
#include <utils/rel.h>
#include <utils/typcache.h>

extern void RegisterCustomScanMethods(const CustomScanMethods* methods);
extern PlannedStmt* standard_planner(Query* parse, const char* query_string,
                                     int32_t cursorOptions, ParamListInfo boundParams);
}

namespace pg {

// Analyze the planned statement and find referenced tables/columns
void analyze_plan(PlannedStmt* plan)
{
    if (plan->permInfos == nullptr) {
        return;
    }
    ListCell* lc = nullptr;
    bool all_tables_are_deeplake = true;
    foreach (lc, plan->permInfos) {
        RTEPermissionInfo* perminfo = (RTEPermissionInfo*)lfirst(lc);
        if (perminfo->relid == InvalidOid) {
            continue;
        }
        auto* table_data = pg::table_storage::instance().get_table_data_if_exists(perminfo->relid);
        if (table_data == nullptr) {
            if (!pg::table_storage::instance().view_exists(perminfo->relid)) {
                pg::query_info::current().set_all_tables_are_deeplake(false);
                all_tables_are_deeplake = false;
            }
            continue;
        }
        table_data->refresh();
        pg::query_info::current().set_is_deeplake_table_referenced(true);
        // Extract columns from selectedCols bitmapset
        if (perminfo->selectedCols == nullptr) {
            continue;
        }
        int32_t col = -1;
        while ((col = bms_next_member(perminfo->selectedCols, col)) >= 0) {
            // bms_next_member returns 0-based index, but AttrNumber is 1-based
            // The bitmapset stores (attnum - FirstLowInvalidHeapAttributeNumber)
            AttrNumber attnum = static_cast<AttrNumber>(col + FirstLowInvalidHeapAttributeNumber);
            if (attnum <= 0) { // Only positive attribute numbers are real columns
                continue;
            }
            auto col_idx = static_cast<int32_t>(attnum - 1);
            if (!table_data->is_column_requested(col_idx)) {
                table_data->set_column_requested(col_idx, true);
                if (!table_data->column_has_streamer(col_idx) && table_data->can_stream_column(col_idx)) {
                    table_data->create_streamer(col_idx, -1);
                }
            }
        }

        // Warm first batches for all streamers in parallel for cold run optimization.
        // This blocks until all first batches are downloaded but overlaps I/O across columns.
        if (pg::eager_batch_prefetch) {
            try {
                table_data->get_streamers().warm_all_streamers();
            } catch (const std::exception& e) {
                elog(WARNING, "Eager batch prefetch failed during analyze_plan: %s", e.what());
            } catch (...) {
                elog(WARNING, "Eager batch prefetch failed during analyze_plan with unknown exception");
            }
        }
    }
    pg::query_info::current().set_all_tables_are_deeplake(all_tables_are_deeplake);

    // Pre-initialize DuckDB connection early so it's ready when query execution starts.
    // This reduces cold run latency by front-loading DuckDB init.
    if (pg::eager_batch_prefetch && pg::query_info::current().is_deeplake_table_referenced()) {
        pg::ensure_duckdb_initialized();
    }
}

} // namespace pg

namespace {

// Custom scan state for direct query execution
struct DeeplakeExecutorState
{
    CustomScanState css;

    // Query execution state
    std::string query_string;
    pg::duckdb_result_holder duckdb_result; // Direct DuckDB results (no conversion!)
    pg::runtime_printer<std::milli> printer;
    TupleDesc output_tupdesc = nullptr;
    std::vector<Oid> output_base_type_oids;
    size_t current_row = 0;
    size_t total_rows = 0;

    DeeplakeExecutorState()
        : css{}
        , printer("DeeplakeExecutorState")
    {
    }

    ~DeeplakeExecutorState()
    {
        // duckdb_result will be automatically cleaned up
    }

    DeeplakeExecutorState(const DeeplakeExecutorState&) = delete;
    DeeplakeExecutorState& operator=(const DeeplakeExecutorState&) = delete;
    DeeplakeExecutorState(DeeplakeExecutorState&&) = delete;
    DeeplakeExecutorState& operator=(DeeplakeExecutorState&&) = delete;

    Oid get_output_base_type_oid(int32_t column_idx) const
    {
        return output_base_type_oids[column_idx];
    }

    int32_t get_output_atttypmod(int32_t column_idx) const
    {
        Form_pg_attribute attr = TupleDescAttr(output_tupdesc, column_idx);
        return attr->atttypmod;
    }
};

// Simple executor state for COUNT(*) fast path
struct CountExecutorState
{
    CustomScanState css{};
    int64_t count_value = 0;
    bool returned = false;
};

// DeepLake executor forward declarations
Node* deeplake_executor_create_scan_state(CustomScan* cscan);
void deeplake_executor_begin_scan(CustomScanState* node, EState* estate, int32_t eflags);
TupleTableSlot* deeplake_executor_exec_scan(CustomScanState* node);
void deeplake_executor_end_scan(CustomScanState* node);
void deeplake_executor_rescan(CustomScanState* node);
void deeplake_executor_explain(CustomScanState* node, List* ancestors, ExplainState* es);

// COUNT(*) executor forward declarations
Node* count_executor_create_scan_state(CustomScan* cscan);
void count_executor_begin_scan(CustomScanState* node, EState* estate, int32_t eflags);
TupleTableSlot* count_executor_exec_scan(CustomScanState* node);
void count_executor_end_scan(CustomScanState* node);
void count_executor_rescan(CustomScanState* node);
void count_executor_explain(CustomScanState* node, List* ancestors, ExplainState* es);

// Custom scan methods
static CustomScanMethods deeplake_executor_scan_methods;
static CustomExecMethods deeplake_executor_exec_methods;

static CustomScanMethods count_executor_scan_methods;
static CustomExecMethods count_executor_exec_methods;

static void init_executor_methods()
{
    static bool inited = false;
    if (inited) {
        return;
    }
    inited = true;

    // Initialize deeplake executor scan methods
    memset(&deeplake_executor_scan_methods, 0, sizeof(deeplake_executor_scan_methods));
    deeplake_executor_scan_methods.CustomName = "DeeplakeExecutor";
    deeplake_executor_scan_methods.CreateCustomScanState = deeplake_executor_create_scan_state;

    // Initialize deeplake executor exec methods
    memset(&deeplake_executor_exec_methods, 0, sizeof(deeplake_executor_exec_methods));
    deeplake_executor_exec_methods.CustomName = "DeeplakeExecutor";
    deeplake_executor_exec_methods.BeginCustomScan = deeplake_executor_begin_scan;
    deeplake_executor_exec_methods.ExecCustomScan = deeplake_executor_exec_scan;
    deeplake_executor_exec_methods.EndCustomScan = deeplake_executor_end_scan;
    deeplake_executor_exec_methods.ReScanCustomScan = deeplake_executor_rescan;
    deeplake_executor_exec_methods.ExplainCustomScan = deeplake_executor_explain;

    // Initialize COUNT(*) executor scan methods
    memset(&count_executor_scan_methods, 0, sizeof(count_executor_scan_methods));
    count_executor_scan_methods.CustomName = "CountExecutor";
    count_executor_scan_methods.CreateCustomScanState = count_executor_create_scan_state;

    // Initialize COUNT(*) executor exec methods
    memset(&count_executor_exec_methods, 0, sizeof(count_executor_exec_methods));
    count_executor_exec_methods.CustomName = "CountExecutor";
    count_executor_exec_methods.BeginCustomScan = count_executor_begin_scan;
    count_executor_exec_methods.ExecCustomScan = count_executor_exec_scan;
    count_executor_exec_methods.EndCustomScan = count_executor_end_scan;
    count_executor_exec_methods.ReScanCustomScan = count_executor_rescan;
    count_executor_exec_methods.ExplainCustomScan = count_executor_explain;
}

// Helper: Convert deeplake sample to PostgreSQL Datum
Datum deeplake_sample_to_datum(
    const nd::array& samples, size_t index, Oid target_type, int32_t attr_typmod, bool& is_null)
{
    try {
        if (!type_is_array(target_type) && nd::dtype_is_numeric(samples.dtype())) {
            return nd::switch_numeric_dtype(samples.dtype(), [&]<typename T>() {
                return pg::utils::pointer_to_datum<T>(samples.data().data(), target_type, attr_typmod, static_cast<int64_t>(index));
            });
        }
        nd::array sample = (samples.dimensions() == 0 ? samples : samples[static_cast<int64_t>(index)]);
        if (sample.is_none()) {
            is_null = true;
            return (Datum)0;
        }
        is_null = false;
        auto datum = pg::utils::nd_to_datum(sample, target_type, attr_typmod);
        // these are non-numeric types so we can compare with zero datum directly
        // ideally nd_to_datum would return 'null' state (or std::optional) instead of (Datum)0 for nulls
        if (datum == (Datum)0) {
            is_null = true;
        }
        return datum;
    } catch (const pg::exception& e) {
        elog(ERROR, "Deeplake Executor: Failed to convert sample to Datum at index %zu: %s", index, e.what());
    }
    return (Datum)0;
}

// Create scan state node
Node* deeplake_executor_create_scan_state(CustomScan* cscan)
{
    DeeplakeExecutorState* state = (DeeplakeExecutorState*)palloc0(sizeof(DeeplakeExecutorState));
    new (state) DeeplakeExecutorState();

    // Initialize the CustomScanState portion
    NodeSetTag(state, T_CustomScanState);
    init_executor_methods();
    state->css.methods = &deeplake_executor_exec_methods;

    // Set up standard scan state
    state->css.ss.ps.type = T_CustomScanState;

    return (Node*)state;
}

// Begin scan - execute the deeplake query
void deeplake_executor_begin_scan(CustomScanState* node, EState* estate, int32_t eflags)
{
    pg::runtime_printer printer("BeginScan DeeplakeExecutor");
    DeeplakeExecutorState* state = (DeeplakeExecutorState*)node;
    CustomScan* cscan = (CustomScan*)node->ss.ps.plan;

    // Initialize the scan state
    state->css.ss.ps.state = estate;
    state->output_tupdesc = ExecTypeFromTL(cscan->scan.plan.targetlist);
    ExecInitScanTupleSlot(estate, &state->css.ss, state->output_tupdesc, &TTSOpsVirtual);
    state->output_base_type_oids.reserve(state->output_tupdesc->natts);
    for (int32_t i = 0; i < state->output_tupdesc->natts; ++i) {
        Form_pg_attribute attr = TupleDescAttr(state->output_tupdesc, i);
        Oid base_typeid = pg::utils::get_base_type(attr->atttypid);
        state->output_base_type_oids.push_back(base_typeid);
    }

    // Extract query string from custom_private
    ASSERT(list_length(cscan->custom_private) >= 1);
    Const* query_const = (Const*)linitial(cscan->custom_private);
    if (query_const && IsA(query_const, Const) && !query_const->constisnull) {
        char* query_str = TextDatumGetCString(query_const->constvalue);
        state->query_string = std::string(query_str);
    }
    ASSERT(!state->query_string.empty());

    // Execute the query through DuckDB and get results directly
    try {
        // Execute SQL query and get DuckDB results without conversion
        state->duckdb_result = pg::execute_sql_query_direct(state->query_string);
        state->total_rows = state->duckdb_result.total_rows;
        state->current_row = 0;

        elog(DEBUG1,
             "DeepLake Executor: Query returned %zu rows with %zu columns (direct from DuckDB)",
             state->total_rows,
             state->duckdb_result.get_column_count());
    } catch (const std::exception& e) {
        elog(ERROR, "DeepLake Executor: Query execution failed: %s", e.what());
    }
}

// Execute scan - return next tuple
TupleTableSlot* deeplake_executor_exec_scan(CustomScanState* node)
{
    DeeplakeExecutorState* state = (DeeplakeExecutorState*)node;
    TupleTableSlot* slot = node->ss.ss_ScanTupleSlot;

    ExecClearTuple(slot);

    if (state->current_row < state->total_rows) {
        // Get the chunk and offset for the current row
        auto [chunk_idx, row_in_chunk] = state->duckdb_result.get_chunk_and_offset(state->current_row);
        void* chunk_ptr = state->duckdb_result.get_chunk_ptr(chunk_idx);
        if (!chunk_ptr) {
            return slot; // Invalid chunk
        }
        auto* chunk = static_cast<duckdb::DataChunk*>(chunk_ptr);

        // Convert each column value directly from DuckDB to PostgreSQL
        for (size_t col_idx = 0; col_idx < chunk->ColumnCount(); ++col_idx) {
            slot->tts_values[col_idx] = pg::duckdb_value_to_pg_datum(chunk->data[col_idx],
                                                                     row_in_chunk,
                                                                     state->get_output_base_type_oid(col_idx),
                                                                     state->get_output_atttypmod(col_idx),
                                                                     slot->tts_isnull[col_idx]);
        }
        ExecStoreVirtualTuple(slot);
        ++state->current_row;
    }

    return slot;
}

// End scan - cleanup
void deeplake_executor_end_scan(CustomScanState* node)
{
    DeeplakeExecutorState* state = (DeeplakeExecutorState*)node;
    state->~DeeplakeExecutorState();
    pfree(state);
}

// Rescan - restart the scan
void deeplake_executor_rescan(CustomScanState* node)
{
    DeeplakeExecutorState* state = (DeeplakeExecutorState*)node;
    state->current_row = 0;
}

// Explain - show query information
void deeplake_executor_explain(CustomScanState* node, List* ancestors, ExplainState* es)
{
    DeeplakeExecutorState* state = (DeeplakeExecutorState*)node;
    ExplainPropertyText("DeepLake Query", state->query_string.c_str(), es);
    if (state->total_rows > 0) {
        ExplainPropertyInteger("Rows", nullptr, static_cast<int64>(state->total_rows), es);
        ExplainPropertyInteger("Chunks", nullptr, static_cast<int64>(state->duckdb_result.get_chunk_count()), es);
    }
}

// ============================================================================
// COUNT(*) Fast Path Executor Implementation
// ============================================================================

// Create scan state for COUNT(*) executor
Node* count_executor_create_scan_state(CustomScan* cscan)
{
    CountExecutorState* state = (CountExecutorState*)palloc0(sizeof(CountExecutorState));
    NodeSetTag(state, T_CustomScanState);
    init_executor_methods();
    state->css.methods = &count_executor_exec_methods;
    state->css.ss.ps.type = T_CustomScanState;
    return (Node*)state;
}

// Begin scan for COUNT(*) executor
void count_executor_begin_scan(CustomScanState* node, EState* estate, int32_t eflags)
{
    CountExecutorState* state = (CountExecutorState*)node;
    CustomScan* cscan = (CustomScan*)node->ss.ps.plan;

    // Initialize scan state
    state->css.ss.ps.state = estate;
    ExecInitScanTupleSlot(estate, &state->css.ss, ExecTypeFromTL(cscan->scan.plan.targetlist), &TTSOpsVirtual);

    // Extract count value from custom_private
    ASSERT(list_length(cscan->custom_private) == 1);
    Const* count_const = (Const*)linitial(cscan->custom_private);
    ASSERT(count_const && IsA(count_const, Const) && !count_const->constisnull);
    state->count_value = DatumGetInt64(count_const->constvalue);
    state->returned = false;

    elog(DEBUG1, "DeepLake COUNT(*) Fast Path: returning %ld", state->count_value);
}

// Execute scan for COUNT(*) executor - return single row with count
TupleTableSlot* count_executor_exec_scan(CustomScanState* node)
{
    CountExecutorState* state = (CountExecutorState*)node;
    TupleTableSlot* slot = node->ss.ss_ScanTupleSlot;

    ExecClearTuple(slot);

    // Return count value as single row, only once
    if (!state->returned) {
        slot->tts_values[0] = Int64GetDatum(state->count_value);
        slot->tts_isnull[0] = false;
        ExecStoreVirtualTuple(slot);
        state->returned = true;
        return slot;
    }

    // Already returned the count
    return slot;
}

// End scan for COUNT(*) executor
void count_executor_end_scan(CustomScanState* node)
{
    // Nothing to clean up
}

// Rescan for COUNT(*) executor
void count_executor_rescan(CustomScanState* node)
{
    CountExecutorState* state = (CountExecutorState*)node;
    state->returned = false;
}

// Explain for COUNT(*) executor
void count_executor_explain(CustomScanState* node, List* ancestors, ExplainState* es)
{
    CountExecutorState* state = (CountExecutorState*)node;
    ExplainPropertyText("DeepLake Optimization", "COUNT(*) Fast Path", es);
    ExplainPropertyInteger("Count", nullptr, state->count_value, es);
}

// ============================================================================

// Create a simple targetlist with Var nodes for the output columns
// This converts expressions to simple column references
List* create_simple_targetlist(List* original_targetlist)
{
    List* new_targetlist = NIL;
    ListCell* lc = nullptr;
    AttrNumber attno = 1;

    foreach (lc, original_targetlist) {
        TargetEntry* tle = (TargetEntry*)lfirst(lc);

        // Create a new Var node representing this output column
        Var* var = makeVar(INDEX_VAR,                       // varno - special value for scan output
                           attno,                           // varattno - column position
                           exprType((Node*)tle->expr),      // vartype
                           exprTypmod((Node*)tle->expr),    // vartypmod
                           exprCollation((Node*)tle->expr), // varcollid
                           0                                // varlevelsup
        );

        // Create a new target entry with the Var
        TargetEntry* new_tle =
            makeTargetEntry((Expr*)var, attno, tle->resname ? pstrdup(tle->resname) : NULL, tle->resjunk);

        new_targetlist = lappend(new_targetlist, new_tle);
        attno++;
    }

    return new_targetlist;
}

} // unnamed namespace

// Register the executor
extern "C" void register_deeplake_executor()
{
    init_executor_methods();
    RegisterCustomScanMethods(&deeplake_executor_scan_methods);
}

// Create a planned statement for direct deeplake execution
extern "C" PlannedStmt* deeplake_create_direct_execution_plan(Query* parse,
                                                              const char* query_string,
                                                              int32_t cursorOptions,
                                                              ParamListInfo boundParams)
{
    // Only handle SELECT queries
    if (parse->commandType != CMD_SELECT || parse->rtable == NIL || list_length(parse->rtable) == 0) {
        return nullptr;
    }

    // Initialize executor methods
    init_executor_methods();

    // Call standard planner to get the output structure
    PlannedStmt* std_plan = standard_planner(parse, query_string, cursorOptions, boundParams);
    pg::analyze_plan(std_plan);
    // <#> marker indicates usage of DeepLake scored index searches - skip direct execution
    if (!pg::query_info::current().are_all_tables_deeplake() || strstr(query_string, "<#>") != nullptr) {
        return std_plan;
    }

    // Fast path for COUNT(*) without WHERE
    if (pg::is_pure_count_star_query(parse)) {
        // Extract table from rtable
        RangeTblEntry* rte = (RangeTblEntry*)linitial(parse->rtable);
        Oid table_id = rte->relid;

        // Get row count from table_data
        if (pg::table_storage::instance().table_exists(table_id)) {
            auto& table_data = pg::table_storage::instance().get_table_data(table_id);
            int64_t row_count = table_data.num_rows();

            // Create a CustomScan node with COUNT(*) executor
            CustomScan* cscan = makeNode(CustomScan);
            cscan->scan.plan.targetlist = create_simple_targetlist(std_plan->planTree->targetlist);
            cscan->scan.plan.qual = NIL;
            cscan->scan.scanrelid = 0;
            cscan->flags = 0;
            cscan->methods = &count_executor_scan_methods;

            // Cost estimates (very cheap!)
            cscan->scan.plan.startup_cost = 0;
            cscan->scan.plan.total_cost = 0.01;
            cscan->scan.plan.plan_rows = 1;
            cscan->scan.plan.plan_width = sizeof(int64_t);

            // Store count value in custom_private
            Const* count_const = makeConst(INT8OID, -1, InvalidOid, sizeof(int64_t),
                                           Int64GetDatum(row_count), false, true);
            cscan->custom_private = list_make1(count_const);

            // Replace plan tree with COUNT(*) executor
            std_plan->planTree = reinterpret_cast<Plan*>(cscan);
            return std_plan;
        }
    }

    // Create a CustomScan node that will execute the entire query in DeepLake
    CustomScan* cscan = makeNode(CustomScan);

    // Create a simple targetlist with Var nodes (no Aggref or complex expressions)
    // This represents the output columns that DeepLake will return
    cscan->scan.plan.targetlist = create_simple_targetlist(std_plan->planTree->targetlist);
    cscan->scan.plan.qual = NIL; // DeepLake handles all filtering
    cscan->scan.scanrelid = 0;
    cscan->flags = 0;
    cscan->methods = &deeplake_executor_scan_methods;

    // Cost estimates
    cscan->scan.plan.startup_cost = std_plan->planTree->startup_cost;
    cscan->scan.plan.total_cost = std_plan->planTree->total_cost;
    cscan->scan.plan.plan_rows = std_plan->planTree->plan_rows;
    cscan->scan.plan.plan_width = std_plan->planTree->plan_width;

    // Store the query string in custom_private
    Const* query_const = makeConst(TEXTOID, -1, InvalidOid, -1, CStringGetTextDatum(query_string), false, false);
    cscan->custom_private = list_make1(query_const);

    // Replace the entire plan tree with our custom scan
    // DeepLake will execute everything including aggregates, joins, sorts, etc.
    std_plan->planTree = reinterpret_cast<Plan*>(cscan);

    return std_plan;
}
