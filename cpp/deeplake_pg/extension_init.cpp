// Include libintl.h first to avoid conflicts with PostgreSQL's gettext macro
#include <libintl.h>

#ifdef __cplusplus
#define typeof __typeof__
extern "C" {
#endif

#include <postgres.h>

#include <catalog/namespace.h>
#include <commands/defrem.h>
#include <commands/vacuum.h>
#include <nodes/nodeFuncs.h>
#include <optimizer/planner.h>
#include <parser/parser.h>
#include <postmaster/bgworker.h>
#include <storage/ipc.h>
#include <tcop/utility.h>
#include <utils/jsonb.h>

#ifdef __cplusplus
} /// extern "C"
#endif

#include "column_statistics.hpp"
#include "deeplake_executor.hpp"
#include "pg_deeplake.hpp"
#include "pg_version_compat.h"
#include "sync_worker.hpp"
#include "table_am.hpp"
#include "table_ddl_lock.hpp"
#include "table_scan.hpp"
#include "table_storage.hpp"
#include "table_version.hpp"

#include "memory_tracker.hpp"
#include "reporter.hpp"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <vector>

// Define GUC variables (declared as extern in utils.hpp)
namespace pg {

bool use_parallel_workers = false;
bool use_deeplake_executor = true;
bool explain_query_before_execute = false;
bool print_runtime_stats = false;
bool support_json_index = false;
bool is_filter_pushdown_enabled = true;
int32_t max_streamable_column_width = 128;
int32_t max_num_threads_for_global_state = std::thread::hardware_concurrency();
bool treat_numeric_as_double = true; // Treat numeric types as double by default
bool print_progress_during_seq_scan = false;
bool use_shared_mem_for_refresh = false;
bool enable_dataset_logging = false; // Enable dataset operation logging for debugging
bool allow_custom_paths = true;      // Allow dataset_path in CREATE TABLE options
bool stateless_enabled = false;      // Enable stateless catalog sync across instances

} // namespace pg

namespace {

bool is_count_star(TargetEntry* node)
{
    if (node == nullptr || node->expr == nullptr || !IsA(node->expr, Aggref)) {
        return false;
    }

    Aggref* agg = (Aggref*)node->expr;
    return ((agg->aggfnoid == F_COUNT_ANY || agg->aggfnoid == F_COUNT_) && (agg->args == NIL || agg->aggstar));
}

void initialize_guc_parameters()
{
    DefineCustomBoolVariable("pg_deeplake.treat_numeric_as_double",
                             "If set to true, numeric values will be treated as double precision.",
                             nullptr,                      // optional long description
                             &pg::treat_numeric_as_double, // linked C variable
                             true,                         // default value
                             PGC_USERSET,                  // context (USERSET, SUSET, etc.)
                             0,                            // flags
                             nullptr,
                             nullptr,
                             nullptr // check_hook, assign_hook, show_hook
    );

    DefineCustomBoolVariable("pg_deeplake.print_progress_during_seq_scan",
                             "Print progress during sequential scan.",
                             nullptr,                             // optional long description
                             &pg::print_progress_during_seq_scan, // linked C variable
                             false,                               // default value
                             PGC_USERSET,                         // context (USERSET, SUSET, etc.)
                             0,                                   // flags
                             nullptr,
                             nullptr,
                             nullptr // check_hook, assign_hook, show_hook
    );

    DefineCustomBoolVariable("pg_deeplake.enable_parallel_workers",
                             "Enable parallel workers for pg_deeplake operations.",
                             nullptr,                   // optional long description
                             &pg::use_parallel_workers, // linked C variable
                             false,                     // default value
                             PGC_USERSET,               // context (USERSET, SUSET, etc.)
                             0,                         // flags
                             nullptr,
                             nullptr,
                             nullptr // check_hook, assign_hook, show_hook
    );

    DefineCustomBoolVariable("pg_deeplake.use_deeplake_executor",
                             "Enable direct execution for pg_deeplake operations.",
                             nullptr,                    // optional long description
                             &pg::use_deeplake_executor, // linked C variable
                             true,                       // default value
                             PGC_USERSET,                // context (USERSET, SUSET, etc.)
                             0,                          // flags
                             nullptr,
                             nullptr,
                             nullptr // check_hook, assign_hook, show_hook
    );

    DefineCustomBoolVariable("pg_deeplake.explain_query_before_execute",
                             "Enable query explanation before execution.",
                             nullptr,                           // optional long description
                             &pg::explain_query_before_execute, // linked C variable
                             false,                             // default value
                             PGC_USERSET,                       // context (USERSET, SUSET, etc.)
                             0,                                 // flags
                             nullptr,
                             nullptr,
                             nullptr // check_hook, assign_hook, show_hook
    );

    DefineCustomBoolVariable("pg_deeplake.print_runtime_stats",
                             "Enable runtime statistics printing for pg_deeplake operations.",
                             nullptr,                  // optional long description
                             &pg::print_runtime_stats, // linked C variable
                             false,                    // default value
                             PGC_USERSET,              // context (USERSET, SUSET, etc.)
                             0,                        // flags
                             nullptr,
                             nullptr,
                             nullptr // check_hook, assign_hook, show_hook
    );

    DefineCustomBoolVariable("pg_deeplake.support_json_index",
                             "Enable support for JSONB index optimizations.",
                             nullptr,                 // optional long description
                             &pg::support_json_index, // linked C variable
                             false,                   // default value
                             PGC_USERSET,             // context (USERSET, SUSET, etc.)
                             0,                       // flags
                             nullptr,
                             nullptr,
                             nullptr // check_hook, assign_hook, show_hook
    );

    DefineCustomBoolVariable("pg_deeplake.is_filter_pushdown_enabled",
                             "Enable filter pushdown optimizations for pg_deeplake tables.",
                             nullptr,                         // optional long description
                             &pg::is_filter_pushdown_enabled, // linked C variable
                             true,                            // default value
                             PGC_USERSET,                     // context (USERSET, SUSET, etc.)
                             0,                               // flags
                             nullptr,
                             nullptr,
                             nullptr // check_hook, assign_hook, show_hook
    );

    DefineCustomIntVariable("pg_deeplake.max_streamable_column_width",
                            "Maximum width (in bytes) for columns to be considered streamable.",
                            nullptr,                          // optional long description
                            &pg::max_streamable_column_width, // linked C variable
                            128,                              // default value (128 bytes)
                            1,                                // min value
                            1024,                             // max value (1 KB)
                            PGC_USERSET,                      // context (USERSET, SUSET, etc.)
                            0,                                // flags
                            nullptr,
                            nullptr,
                            nullptr // check_hook, assign_hook, show_hook
    );

    DefineCustomIntVariable("pg_deeplake.max_num_threads_for_global_state",
                            "Maximum number of threads for global state operations.",
                            nullptr,                               // optional long description
                            &pg::max_num_threads_for_global_state, // linked C variable
                            base::system_report::cpu_cores(),      // default value
                            1,                                     // min value
                            base::system_report::cpu_cores(),      // max value
                            PGC_USERSET,                           // context (USERSET, SUSET, etc.)
                            0,                                     // flags
                            nullptr,
                            nullptr,
                            nullptr // check_hook, assign_hook, show_hook
    );

    DefineCustomBoolVariable("pg_deeplake.use_shared_mem_for_refresh",
                             "Use shared memory to detect whether dataset was refreshed or not.",
                             "Note: Should have same value across all instances in a cluster. "
                             "Enabling this option allows the system to use shared memory "
                             "for detecting dataset refreshes, which can improve performance but may "
                             "have implications on concurrency. "
                             "It make sense to disable this for OLTP workloads.",
                             &pg::use_shared_mem_for_refresh, // linked C variable
                             true,                            // default value
                             PGC_USERSET,                     // context (USERSET, SUSET, etc.)
                             0,                               // flags
                             nullptr,
                             nullptr,
                             nullptr // check_hook, assign_hook, show_hook
    );

    DefineCustomBoolVariable("deeplake.allow_custom_paths",
                             "Allow custom dataset paths via USING deeplake WITH (dataset_path=...).",
                             "If disabled, dataset_path options are rejected and tables must use deeplake.root_path.",
                             &pg::allow_custom_paths,
                             true,
                             PGC_USERSET,
                             0,
                             nullptr,
                             nullptr,
                             nullptr);

    DefineCustomBoolVariable("deeplake.stateless_enabled",
                             "Enable stateless catalog for multi-instance sync.",
                             "When enabled, table metadata is written to the shared catalog in storage, "
                             "allowing multiple PostgreSQL instances to share the same tables. "
                             "This adds latency for remote storage (S3, GCS) due to catalog sync operations.",
                             &pg::stateless_enabled,
                             false,
                             PGC_USERSET,
                             0,
                             nullptr,
                             nullptr,
                             nullptr);

    DefineCustomBoolVariable("pg_deeplake.enable_dataset_logging",
                             "Enable operation logging for deeplake datasets.",
                             "When enabled, all dataset operations (append_row, update_row, delete_row, etc.) "
                             "are logged to debug_logs/{session_id}/{timestamp}.json files in the dataset storage. "
                             "This can be useful for debugging and auditing purposes. "
                             "Note: This may have a performance impact as each operation is logged asynchronously.",
                             &pg::enable_dataset_logging, // linked C variable
                             false,                       // default value
                             PGC_USERSET,                 // context (USERSET, SUSET, etc.)
                             0,                           // flags
                             nullptr,
                             nullptr,
                             nullptr // check_hook, assign_hook, show_hook
    );

    // Sync worker GUC variables for stateless multi-instance support
    DefineCustomIntVariable("deeplake.sync_interval_ms",
                            "Interval between catalog sync checks in milliseconds.",
                            "The background sync worker polls the catalog version at this interval. "
                            "When the version changes, tables are synced from the shared catalog.",
                            &deeplake_sync_interval_ms, // linked C variable
                            2000,                       // default value (2 seconds)
                            100,                        // min value
                            60000,                      // max value (1 minute)
                            PGC_SIGHUP,                 // context - reloadable
                            GUC_UNIT_MS,                // flags
                            nullptr,
                            nullptr,
                            nullptr // check_hook, assign_hook, show_hook
    );


    // Initialize PostgreSQL memory tracking
    pg::memory_tracker::initialize_guc_parameters();

    pg::session_credentials::initialize_guc();
}

/// @name Json path extraction and transformation
/// @description: Helper to extract path from nested -> and ->> operators
/// @note: Under GUC param pg_deeplake.support_json_index (default: false)
/// @{

void extract_jsonb_path(Node* expr, std::vector<std::string>& path, Node** base_col)
{
    if (IsA(expr, OpExpr)) {
        OpExpr* op = (OpExpr*)expr;
        char* opname = get_opname(op->opno);

        if (opname && (strcmp(opname, "->") == 0 || strcmp(opname, "->>") == 0)) {
            Node* left = (Node*)linitial(op->args);
            Node* right = (Node*)lsecond(op->args);

            // Recursively extract path from left side
            extract_jsonb_path(left, path, base_col);

            // Add current key to path
            if (IsA(right, Const)) {
                Const* kc = (Const*)right;
                if (kc->consttype == TEXTOID && !kc->constisnull) {
                    char* key = text_to_cstring(DatumGetTextPP(kc->constvalue));
                    path.push_back(std::string(key));
                    pfree(key);
                }
            }
        } else {
            *base_col = expr;
        }

        if (opname)
            pfree(opname);
    } else {
        *base_col = expr;
    }
}

// Build nested JSON from path: ["a", "b", "c"] with value "v" -> {"a": {"b": {"c": "v"}}}
std::string build_nested_json(const std::vector<std::string>& path, const std::string& value)
{
    if (path.empty())
        return "";

    StringInfoData json;
    initStringInfo(&json);

    // Open braces for each level
    for (size_t i = 0; i < path.size(); i++) {
        appendStringInfo(&json, "{\"%s\":", path[i].c_str());
    }

    // Add value
    appendStringInfo(&json, "\"%s\"", value.c_str());

    // Close braces
    for (size_t i = 0; i < path.size(); i++) {
        appendStringInfoChar(&json, '}');
    }

    std::string result(json.data);
    pfree(json.data);
    return result;
}

// Transform ->> expressions into @> containment for JSONB index usage
void transform_jsonb_arrow_quals(Node** nodeptr)
{
    if (!nodeptr || !*nodeptr)
        return;
    Node* node = *nodeptr;

    if (IsA(node, BoolExpr)) {
        BoolExpr* b = (BoolExpr*)node;
        ListCell* lc = nullptr;
        foreach (lc, b->args) {
            transform_jsonb_arrow_quals((Node**)&lfirst(lc));
        }
    } else if (IsA(node, OpExpr)) {
        OpExpr* op = (OpExpr*)node;
        char* opname = get_opname(op->opno);

        // Look for: (data -> 'a' -> 'b' ->> 'c') = 'value' or (data ->> 'key') = 'value'
        // Transform to: data @> '{"a": {"b": {"c": "value"}}}'::jsonb
        if (opname && strcmp(opname, "=") == 0 && list_length(op->args) == 2) {
            Node* left = (Node*)linitial(op->args);
            Node* right = (Node*)lsecond(op->args);

            // Check if left side has -> or ->> operators and right is a constant
            if (IsA(left, OpExpr) && IsA(right, Const)) {
                // Extract the full path from nested operators
                std::vector<std::string> path;
                Node* base_col = nullptr;
                extract_jsonb_path(left, path, &base_col);

                if (base_col && !path.empty()) {
                    Oid col_type = exprType(base_col);
                    Const* val = (Const*)right;

                    if (col_type == JSONBOID && !val->constisnull && val->consttype == TEXTOID) {
                        // Build nested JSON
                        char* vs = text_to_cstring(DatumGetTextPP(val->constvalue));
                        std::string json_str = build_nested_json(path, vs);

                        if (!json_str.empty()) {
                            // Convert to JSONB
                            Jsonb* jb =
                                DatumGetJsonbP(DirectFunctionCall1(jsonb_in, CStringGetDatum(json_str.c_str())));

                            // Create JSONB constant
                            Const* jc = makeNode(Const);
                            jc->consttype = JSONBOID;
                            jc->consttypmod = -1;
                            jc->constcollid = InvalidOid;
                            jc->constlen = -1;
                            jc->constvalue = JsonbPGetDatum(jb);
                            jc->constisnull = false;
                            jc->constbyval = false;
                            jc->location = val->location;

                            // Look up @> operator
                            Oid cop = OpernameGetOprid(list_make1(makeString(pstrdup("@>"))), JSONBOID, JSONBOID);
                            if (OidIsValid(cop)) {
                                // Create new @> operator expression
                                OpExpr* newop = makeNode(OpExpr);
                                newop->opno = cop;
                                newop->opfuncid = get_opcode(cop);
                                newop->opresulttype = BOOLOID;
                                newop->opretset = false;
                                newop->opcollid = InvalidOid;
                                newop->inputcollid = InvalidOid;
                                newop->args = list_make2(copyObject(base_col), jc);
                                newop->location = op->location;
                                *nodeptr = (Node*)newop;

                                base::log_debug(base::log_channel::index,
                                                "Transformed JSONB path (depth {}) to @> containment",
                                                path.size());
                            }
                        }
                        pfree(vs);
                    }
                }
            }
        }
        if (opname)
            pfree(opname);
    }
}

/// @}

} // namespace

#ifdef __cplusplus
extern "C" {
#endif

PG_MODULE_MAGIC;

PG_FUNCTION_INFO_V1(create_deeplake_table);
PG_FUNCTION_INFO_V1(deeplake_tableam_handler);

static ExecutorStart_hook_type prev_executor_start = nullptr;
static ExecutorRun_hook_type prev_executor_run = nullptr;
static ExecutorEnd_hook_type prev_executor_end = nullptr;
static ProcessUtility_hook_type prev_process_utility_hook = nullptr;
static planner_hook_type prev_planner_hook = nullptr;
typedef void (*set_rel_pathlist_hook_type)(PlannerInfo* root, RelOptInfo* rel, Index rti, RangeTblEntry* rte);
extern PGDLLIMPORT set_rel_pathlist_hook_type set_rel_pathlist_hook;
static set_rel_pathlist_hook_type prev_set_rel_pathlist_hook = nullptr;
static shmem_request_hook_type prev_shmem_request_hook = nullptr;
static shmem_startup_hook_type prev_shmem_startup_hook = nullptr;

/// @name Callbacks for executor hooks
/// @{

static void deeplake_shmem_request()
{
    if (prev_shmem_request_hook) {
        prev_shmem_request_hook();
    }

    // Request shared memory for table version tracking
    RequestAddinShmemSpace(pg::table_version_tracker::get_shmem_size());
    RequestNamedLWLockTranche("deeplake_versions", 1);

    // Request shared memory for table DDL lock
    RequestAddinShmemSpace(pg::table_ddl_lock::get_shmem_size());
    RequestNamedLWLockTranche("deeplake_table_ddl", 1);
}

static void deeplake_shmem_startup()
{
    if (prev_shmem_startup_hook) {
        prev_shmem_startup_hook();
    }

    pg::table_version_tracker::initialize();
    pg::table_ddl_lock::initialize();
}

static void process_utility(PlannedStmt* pstmt,
                            const char* queryString,
                            bool readOnlyTree,
                            ProcessUtilityContext context,
                            ParamListInfo params,
                            QueryEnvironment* queryEnv,
                            DestReceiver* dest,
                            QueryCompletion* completionTag)
{
    pg::runtime_printer printer("Process Utility Hook");
    pg::init_deeplake();
    if (nodeTag(pstmt->utilityStmt) == T_DropStmt) {
        DropStmt* stmt = (DropStmt*)pstmt->utilityStmt;
        if (stmt->removeType == OBJECT_EXTENSION) {
            // Extension is being dropped, clean up all tables and indexes
            pg::pg_index::clear();
            pg::table_storage::instance().clear();
        } else if (stmt->removeType == OBJECT_INDEX) {
            ListCell* lc = nullptr;
            foreach (lc, stmt->objects) {
                List* object = (List*)lfirst(lc);
                const char* index_name = strVal(linitial(object));
                if (!pg::pg_index::has_indexes()) {
                    pg::load_index_metadata();
                }
                pg::pg_index::erase_info(index_name);
                pg::erase_indexer_data(std::string{}, std::string{}, index_name);
            }
        } else if (stmt->removeType == OBJECT_TABLE) {
            ListCell* lc = nullptr;
            foreach (lc, stmt->objects) {
                List* table_name_list = (List*)lfirst(lc);
                std::string table_name;
                if (list_length(table_name_list) == 2) {
                    auto* sname = strVal(linitial(table_name_list));
                    auto* tname = strVal(lsecond(table_name_list));
                    table_name = std::string(sname) + "." + std::string(tname);
                } else if (list_length(table_name_list) == 1) {
                    table_name = std::string("public.") + strVal(linitial(table_name_list));
                }
                // Handle both index and table cleanup
                if (pg::pg_index::has_index_created_on_table(table_name)) {
                    pg::pg_index::erase_table_info(table_name);
                    pg::erase_indexer_data(table_name, std::string{}, std::string{});
                }
                pg::table_storage::instance().drop_table(table_name);
            }
        } else if (stmt->removeType == OBJECT_VIEW) {
            ListCell* lc = nullptr;
            foreach (lc, stmt->objects) {
                List* view_name_list = (List*)lfirst(lc);
                std::string view_name;
                if (list_length(view_name_list) == 2) {
                    auto* sname = strVal(linitial(view_name_list));
                    auto* vname = strVal(lsecond(view_name_list));
                    view_name = std::string(sname) + "." + std::string(vname);
                } else if (list_length(view_name_list) == 1) {
                    view_name = std::string(strVal(linitial(view_name_list)));
                }
                pg::table_storage::instance().erase_view(view_name);
            }
        } else if (stmt->removeType == OBJECT_SCHEMA) {
            ListCell* lc = nullptr;
            foreach (lc, stmt->objects) {
                String* schema_name_str = (String*)lfirst(lc);
                if (schema_name_str == nullptr || !IsA(schema_name_str, String)) {
                    continue;
                }
                const char* schema_name = strVal(schema_name_str);
                if (schema_name == nullptr) {
                    continue;
                }
                const char* query = "SELECT nspname, relname "
                                    "FROM pg_class c "
                                    "JOIN pg_namespace n ON c.relnamespace = n.oid "
                                    "WHERE c.relkind = 'r' AND n.nspname = $1";
                pg::utils::spi_connector connector;
                Oid argtypes[1] = {TEXTOID};
                Datum values[1];
                values[0] = CStringGetTextDatum(schema_name);
                if (SPI_execute_with_args(query, 1, argtypes, values, nullptr, true, 0) == SPI_OK_SELECT) {
                    for (auto i = 0; i < SPI_processed; ++i) {
                        const char* sname = SPI_getvalue(SPI_tuptable->vals[i], SPI_tuptable->tupdesc, 1);
                        const char* tname = SPI_getvalue(SPI_tuptable->vals[i], SPI_tuptable->tupdesc, 2);
                        if (sname != nullptr && tname != nullptr) {
                            const std::string table_name = (std::string(sname) + "." + std::string(tname));
                            // Handle both index and table cleanup
                            if (pg::pg_index::has_index_created_on_table(table_name)) {
                                pg::pg_index::erase_table_info(table_name);
                                pg::erase_indexer_data(table_name, std::string{}, std::string{});
                            }
                            pg::table_storage::instance().drop_table(table_name);
                        }
                    }
                }
            }
        } else if (stmt->removeType == OBJECT_DATABASE) {
            const char* query = "SELECT nspname, relname "
                                "FROM pg_class c "
                                "JOIN pg_namespace n ON c.relnamespace = n.oid "
                                "WHERE c.relkind = 'r'";
            pg::utils::spi_connector connector;
            if (SPI_execute(query, true, 0) == SPI_OK_SELECT) {
                for (auto i = 0; i < SPI_processed; ++i) {
                    char* sname = SPI_getvalue(SPI_tuptable->vals[i], SPI_tuptable->tupdesc, 1);
                    char* tname = SPI_getvalue(SPI_tuptable->vals[i], SPI_tuptable->tupdesc, 2);
                    if (sname != nullptr && tname != nullptr) {
                        const std::string table_name = (std::string(sname) + "." + std::string(tname));
                        pg::pg_index::erase_table_info(table_name);
                        pg::erase_indexer_data(table_name, std::string{}, std::string{});
                    }
                }
            }
        }
    }
    if (IsA(pstmt->utilityStmt, AlterTableStmt)) {
        AlterTableStmt* stmt = (AlterTableStmt*)pstmt->utilityStmt;

        /// Extract table name
        RangeVar* rel = stmt->relation;

        // Get the actual relation OID to resolve the proper schema name (handles custom schemas via search_path)
        Oid rel_oid = RangeVarGetRelid(rel, NoLock, false);
        Relation temp_rel = RelationIdGetRelation(rel_oid);
        std::string table_name;

        if (RelationIsValid(temp_rel)) {
            Oid nspid = RelationGetNamespace(temp_rel);
            char* nspname = get_namespace_name(nspid);
            table_name = std::string(nspname ? nspname : "public") + "." + RelationGetRelationName(temp_rel);
            if (nspname) {
                pfree(nspname);
            }
            RelationClose(temp_rel);
        } else {
            // Fallback to RangeVar if relation is invalid (shouldn't happen)
            const std::string schema_name = (rel->schemaname != nullptr ? rel->schemaname : "public");
            table_name = schema_name + "." + rel->relname;
        }

        auto* td = pg::table_storage::instance().get_table_data_if_exists(table_name);
        if (td != nullptr) {
            ListCell* lc = nullptr;
            foreach (lc, stmt->cmds) {
                AlterTableCmd* cmd = (AlterTableCmd*)lfirst(lc);
                if (cmd->subtype == AT_DropColumn) {
                    std::string column_name = cmd->name;
                    pg::pg_index::erase_column_info(table_name, column_name);
                    pg::erase_indexer_data(table_name, column_name, std::string{});
                    td->get_dataset()->remove_column(column_name);
                }
            }
            td->commit();
            // Note: We don't reload table_data here because the TupleDesc hasn't been updated yet.
            // The reload will happen in the POST-AlterTable section below after standard_ProcessUtility.
        }
    }

    if (IsA(pstmt->utilityStmt, CreateStmt)) {
        CreateStmt* stmt = (CreateStmt*)pstmt->utilityStmt;
        elog(DEBUG1, "CreateStmt: %s", stmt->relation->relname);
        elog(DEBUG1, "stmt->options: %p", stmt->options);
        elog(DEBUG1, "stmt->accessMethod: %s", stmt->accessMethod);
        const bool deeplake_table = (stmt->accessMethod != nullptr && std::strcmp(stmt->accessMethod, "deeplake") == 0);
        if (deeplake_table && stmt->options != nullptr) {
            List* new_options = NIL;
            ListCell* lc = nullptr;
            foreach (lc, stmt->options) {
                DefElem* def = (DefElem*)lfirst(lc);
                if (def->arg != nullptr && std::strcmp(def->defname, pg::dataset_path_option_name) == 0) {
                    const char* ds_path = defGetString(def);
                    pg::table_options::current().set_dataset_path(ds_path);
                    elog(DEBUG1, "ds_path: %s", ds_path);
                    continue;
                }
                new_options = lappend(new_options, def);
            }
            stmt->options = new_options;
        }
    }

    std::optional<pg::utils::parallel_workers_switcher> switcher;
    if (IsA(pstmt->utilityStmt, IndexStmt)) {
        IndexStmt* stmt = (IndexStmt*)pstmt->utilityStmt;
        const Oid rel_id = RangeVarGetRelid(stmt->relation, NoLock, false);
        if (pg::table_storage::instance().table_exists(rel_id)) {
            switcher.emplace();
        }
    }

    if (prev_process_utility_hook != nullptr) {
        prev_process_utility_hook(pstmt, queryString, readOnlyTree, context, params, queryEnv, dest, completionTag);
    } else {
        standard_ProcessUtility(pstmt, queryString, readOnlyTree, context, params, queryEnv, dest, completionTag);
    }

    // Post-process ALTER TABLE ADD COLUMN to add column to deeplake dataset
    if (IsA(pstmt->utilityStmt, AlterTableStmt)) {
        AlterTableStmt* stmt = (AlterTableStmt*)pstmt->utilityStmt;
        RangeVar* rel = stmt->relation;

        // Get the actual relation OID to resolve the proper schema name (handles custom schemas via search_path)
        Oid rel_oid = RangeVarGetRelid(rel, NoLock, false);
        Relation temp_rel = RelationIdGetRelation(rel_oid);
        std::string table_name;

        if (RelationIsValid(temp_rel)) {
            Oid nspid = RelationGetNamespace(temp_rel);
            char* nspname = get_namespace_name(nspid);
            table_name = std::string(nspname ? nspname : "public") + "." + RelationGetRelationName(temp_rel);
            if (nspname) {
                pfree(nspname);
            }
            RelationClose(temp_rel);
        } else {
            // Fallback to RangeVar if relation is invalid (shouldn't happen)
            const std::string schema_name = (rel->schemaname != nullptr ? rel->schemaname : "public");
            table_name = schema_name + "." + rel->relname;
        }

        auto* td = pg::table_storage::instance().get_table_data_if_exists(table_name);

        if (td != nullptr) {
            ListCell* lc = nullptr;
            foreach (lc, stmt->cmds) {
                AlterTableCmd* cmd = (AlterTableCmd*)lfirst(lc);
                if (cmd->subtype == AT_AddColumn && cmd->def != nullptr) {
                    // Column has been added to PostgreSQL catalog, now add to deeplake
                    ColumnDef* coldef = (ColumnDef*)cmd->def;
                    const char* column_name = coldef->colname;

                    // Get the relation to query the new column's type from catalog
                    Relation relation = RelationIdGetRelation(rel_oid);
                    if (RelationIsValid(relation)) {
                        TupleDesc tupdesc = RelationGetDescr(relation);

                        // Find the newly added column in the tuple descriptor
                        for (int i = 0; i < tupdesc->natts; i++) {
                            Form_pg_attribute attr = TupleDescAttr(tupdesc, i);
                            if (strcmp(NameStr(attr->attname), column_name) == 0) {
                                // Found the new column, add it to deeplake dataset
                                Oid base_typeid = attr->atttypid;

                                // Resolve domain types to their base type
                                HeapTuple type_tuple = SearchSysCache1(TYPEOID, ObjectIdGetDatum(base_typeid));
                                if (HeapTupleIsValid(type_tuple)) {
                                    Form_pg_type type_form = (Form_pg_type)GETSTRUCT(type_tuple);
                                    if (type_form->typtype == TYPTYPE_DOMAIN) {
                                        base_typeid = type_form->typbasetype;
                                    }
                                    ReleaseSysCache(type_tuple);
                                }

                                try {
                                    auto ds = td->get_dataset();

                                    // Map PostgreSQL type to DeepLake type (same logic as create_table)
                                    switch (base_typeid) {
                                    case BOOLOID:
                                        ds->add_column(column_name, nd::type::scalar(nd::dtype::boolean));
                                        break;
                                    case INT2OID:
                                        ds->add_column(column_name, nd::type::scalar(nd::dtype::int16));
                                        break;
                                    case INT4OID:
                                    case DATEOID:
                                        ds->add_column(column_name, nd::type::scalar(nd::dtype::int32));
                                        break;
                                    case TIMEOID:
                                    case TIMESTAMPOID:
                                    case TIMESTAMPTZOID:
                                    case INT8OID:
                                        ds->add_column(column_name, nd::type::scalar(nd::dtype::int64));
                                        break;
                                    case FLOAT4OID:
                                        ds->add_column(column_name, nd::type::scalar(nd::dtype::float32));
                                        break;
                                    case NUMERICOID: {
                                        const int32_t typmod = attr->atttypmod;
                                        if (typmod >= 0) {
                                            const int32_t precision = ((typmod - VARHDRSZ) >> 16) & 0xFFFF;
                                            if (precision > 15) {
                                                const int32_t scale = (typmod - VARHDRSZ) & 0xFFFF;
                                                elog(WARNING,
                                                     "Column '%s' has type NUMERIC(%d, %d), which may lose precision "
                                                     "as it is stored as FLOAT64.",
                                                     column_name,
                                                     precision,
                                                     scale);
                                            }
                                        }
                                    }
                                    case FLOAT8OID:
                                        ds->add_column(column_name, nd::type::scalar(nd::dtype::float64));
                                        break;
                                    case CHAROID:
                                    case BPCHAROID:
                                    case VARCHAROID: {
                                        const int32_t typmod = attr->atttypmod;
                                        if (typmod == VARHDRSZ + 1) {
                                            ds->add_column(
                                                column_name,
                                                deeplake_core::type::generic(nd::type::scalar(nd::dtype::int8)));
                                        } else {
                                            ds->add_column(column_name,
                                                           deeplake_core::type::text(codecs::compression::null));
                                        }
                                        break;
                                    }
                                    case UUIDOID:
                                    case TEXTOID:
                                        ds->add_column(column_name,
                                                       deeplake_core::type::text(codecs::compression::null));
                                        break;
                                    case JSONOID:
                                    case JSONBOID:
                                        ds->add_column(column_name, deeplake_core::type::dict());
                                        break;
                                    case BYTEAOID:
                                        ds->add_column(column_name,
                                                       deeplake_core::type::generic(nd::type::scalar(nd::dtype::byte)));
                                        break;
                                    case INT2ARRAYOID: {
                                        int32_t ndims = (attr->attndims > 0) ? attr->attndims : 1;
                                        if (ndims > 255) {
                                            elog(ERROR,
                                                 "Column '%s' has unsupported type SMALLINT[] with %d dimensions (max "
                                                 "255)",
                                                 column_name,
                                                 ndims);
                                        }
                                        if (ndims == 1) {
                                            ds->add_column(column_name,
                                                           deeplake_core::type::embedding(0, nd::dtype::int16));
                                        } else {
                                            ds->add_column(
                                                column_name,
                                                deeplake_core::type::generic(nd::type::array(nd::dtype::int16, ndims)));
                                        }
                                        break;
                                    }
                                    case INT4ARRAYOID: {
                                        int32_t ndims = (attr->attndims > 0) ? attr->attndims : 1;
                                        if (ndims > 255) {
                                            elog(ERROR,
                                                 "Column '%s' has unsupported type INT[] with %d dimensions (max 255)",
                                                 column_name,
                                                 ndims);
                                        }
                                        if (ndims == 1) {
                                            ds->add_column(column_name,
                                                           deeplake_core::type::embedding(0, nd::dtype::int32));
                                        } else {
                                            ds->add_column(
                                                column_name,
                                                deeplake_core::type::generic(nd::type::array(nd::dtype::int32, ndims)));
                                        }
                                        break;
                                    }
                                    case INT8ARRAYOID: {
                                        int32_t ndims = (attr->attndims > 0) ? attr->attndims : 1;
                                        if (ndims > 255) {
                                            elog(ERROR,
                                                 "Column '%s' has unsupported type BIGINT[] with %d dimensions (max "
                                                 "255)",
                                                 column_name,
                                                 ndims);
                                        }
                                        if (ndims == 1) {
                                            ds->add_column(column_name,
                                                           deeplake_core::type::embedding(0, nd::dtype::int64));
                                        } else {
                                            ds->add_column(
                                                column_name,
                                                deeplake_core::type::generic(nd::type::array(nd::dtype::int64, ndims)));
                                        }
                                        break;
                                    }
                                    case FLOAT4ARRAYOID: {
                                        int32_t ndims = (attr->attndims > 0) ? attr->attndims : 1;
                                        if (ndims > 255) {
                                            elog(ERROR,
                                                 "Column '%s' has unsupported type REAL[] with %d dimensions (max 255)",
                                                 column_name,
                                                 ndims);
                                        }
                                        if (ndims == 1) {
                                            ds->add_column(column_name,
                                                           deeplake_core::type::embedding(0, nd::dtype::float32));
                                        } else {
                                            ds->add_column(column_name,
                                                           deeplake_core::type::generic(
                                                               nd::type::array(nd::dtype::float32, ndims)));
                                        }
                                        break;
                                    }
                                    case FLOAT8ARRAYOID: {
                                        int32_t ndims = (attr->attndims > 0) ? attr->attndims : 1;
                                        if (ndims > 255) {
                                            elog(ERROR,
                                                 "Column '%s' has unsupported type DOUBLE PRECISION[] with %d "
                                                 "dimensions (max 255)",
                                                 column_name,
                                                 ndims);
                                        }
                                        if (ndims == 1) {
                                            ds->add_column(column_name,
                                                           deeplake_core::type::embedding(0, nd::dtype::float64));
                                        } else {
                                            ds->add_column(column_name,
                                                           deeplake_core::type::generic(
                                                               nd::type::array(nd::dtype::float64, ndims)));
                                        }
                                        break;
                                    }
                                    case BYTEAARRAYOID: {
                                        if (attr->attndims > 1) {
                                            elog(ERROR,
                                                 "Column '%s' has unsupported type BYTEA[] with %d dimensions",
                                                 column_name,
                                                 attr->attndims);
                                        }
                                        ds->add_column(column_name,
                                                       deeplake_core::type::generic(
                                                           nd::type::array(nd::dtype::byte, attr->attndims)));
                                        break;
                                    }
                                    case VARCHARARRAYOID:
                                    case TEXTARRAYOID: {
                                        if (attr->attndims > 1) {
                                            elog(ERROR,
                                                 "Column '%s' has unsupported type TEXT[] with %d dimensions",
                                                 column_name,
                                                 attr->attndims);
                                        }
                                        ds->add_column(column_name,
                                                       deeplake_core::type::generic(
                                                           nd::type::array(nd::dtype::string, attr->attndims)));
                                        break;
                                    }
                                    default: {
                                        const char* tname = format_type_with_typemod(attr->atttypid, attr->atttypmod);
                                        elog(ERROR,
                                             "ALTER TABLE ADD COLUMN: Column '%s' has unsupported type '%s' (OID %u, "
                                             "base OID %u)",
                                             column_name,
                                             tname,
                                             attr->atttypid,
                                             base_typeid);
                                    }
                                    }

                                    // Commit the change to deeplake dataset
                                    td->commit();
                                    elog(INFO,
                                         "Added column '%s' to deeplake dataset for table '%s'",
                                         column_name,
                                         table_name.c_str());

                                    // Invalidate cached table data to force reload in current session
                                    RelationClose(relation);
                                    pg::table_storage::instance().erase_table(table_name);
                                    pg::table_storage::instance().mark_metadata_stale();
                                    return; // Exit early after erasing table

                                } catch (const base::exception& e) {
                                    RelationClose(relation);
                                    ereport(ERROR,
                                            (errcode(ERRCODE_INTERNAL_ERROR),
                                             errmsg("Failed to add column '%s' to deeplake dataset: %s",
                                                    column_name,
                                                    e.what())));
                                }
                                break;
                            }
                        }
                        RelationClose(relation);
                    }
                } else if (cmd->subtype == AT_DropColumn) {
                    // Column has been dropped from PostgreSQL catalog, now reload table_data
                    // to pick up the updated TupleDesc (with the column marked as dropped)
                    pg::table_storage::instance().erase_table(table_name);
                    pg::table_storage::instance().mark_metadata_stale();
                    elog(INFO, "Reloaded table_data after DROP COLUMN for table '%s'", table_name.c_str());
                    return; // Exit early after reloading table
                }
            }
        }
    }

    // Post-process RENAME COLUMN to rename column in deeplake dataset
    if (IsA(pstmt->utilityStmt, RenameStmt)) {
        RenameStmt* stmt = (RenameStmt*)pstmt->utilityStmt;
        if (stmt->renameType == OBJECT_COLUMN && stmt->relation != nullptr) {
            RangeVar* rel = stmt->relation;

            // Get the actual relation OID to resolve the proper schema name (handles custom schemas via search_path)
            Oid rel_oid = RangeVarGetRelid(rel, NoLock, false);
            Relation temp_rel = RelationIdGetRelation(rel_oid);
            std::string table_name;

            if (RelationIsValid(temp_rel)) {
                Oid nspid = RelationGetNamespace(temp_rel);
                char* nspname = get_namespace_name(nspid);
                table_name = std::string(nspname ? nspname : "public") + "." + RelationGetRelationName(temp_rel);
                if (nspname) {
                    pfree(nspname);
                }
                RelationClose(temp_rel);
            } else {
                // Fallback to RangeVar if relation is invalid (shouldn't happen)
                const std::string schema_name = (rel->schemaname != nullptr ? rel->schemaname : "public");
                table_name = schema_name + "." + rel->relname;
            }

            auto* td = pg::table_storage::instance().get_table_data_if_exists(table_name);

            if (td != nullptr && stmt->subname != nullptr && stmt->newname != nullptr) {
                const char* old_column_name = stmt->subname;
                const char* new_column_name = stmt->newname;

                try {
                    auto ds = td->get_dataset();

                    // Rename column in deeplake dataset
                    ds->rename_column(old_column_name, new_column_name);

                    // Commit the change to deeplake dataset
                    td->commit();
                    elog(INFO,
                         "Renamed column '%s' to '%s' in deeplake dataset for table '%s'",
                         old_column_name,
                         new_column_name,
                         table_name.c_str());

                    // Invalidate cached table data to force reload in current session
                    pg::table_storage::instance().erase_table(table_name);
                    pg::table_storage::instance().mark_metadata_stale();

                } catch (const base::exception& e) {
                    ereport(ERROR,
                            (errcode(ERRCODE_INTERNAL_ERROR),
                             errmsg("Failed to rename column '%s' to '%s' in deeplake dataset: %s",
                                    old_column_name,
                                    new_column_name,
                                    e.what())));
                }
            }
        }
    }

    if (IsA(pstmt->utilityStmt, CopyStmt)) {
        CopyStmt* copy_stmt = (CopyStmt*)pstmt->utilityStmt;
        if (copy_stmt->relation) {
            // Build the qualified table name
            // Get the actual relation OID to resolve the proper schema name (handles custom schemas via search_path)
            Oid rel_oid = RangeVarGetRelid(copy_stmt->relation, NoLock, false);
            Relation temp_rel = RelationIdGetRelation(rel_oid);
            std::string table_name;

            if (RelationIsValid(temp_rel)) {
                Oid nspid = RelationGetNamespace(temp_rel);
                char* nspname = get_namespace_name(nspid);
                table_name = std::string(nspname ? nspname : "public") + "." + RelationGetRelationName(temp_rel);
                if (nspname) {
                    pfree(nspname);
                }
                RelationClose(temp_rel);
            } else {
                // Fallback to RangeVar if relation is invalid (shouldn't happen)
                const char* schema = copy_stmt->relation->schemaname ? copy_stmt->relation->schemaname : "public";
                const char* table = copy_stmt->relation->relname;
                table_name = std::string(schema) + "." + table;
            }

            // If this is a deeplake table, flush/commit
            auto* td = pg::table_storage::instance().get_table_data_if_exists(table_name);
            if (td) {
                if (!td->flush()) {
                    ereport(ERROR, (errmsg("Failed to flush inserts after COPY")));
                }
            }
        }
    }

    // Handle ANALYZE statement - inject pre-computed DeepLake statistics
    // This runs after PostgreSQL's standard ANALYZE to override sampled statistics
    // with more accurate pre-computed statistics from DeepLake's incremental computation
    if (IsA(pstmt->utilityStmt, VacuumStmt)) {
        VacuumStmt* vstmt = (VacuumStmt*)pstmt->utilityStmt;

        // Check if this is an ANALYZE operation (is_vacuumcmd=false means ANALYZE)
        if (!vstmt->is_vacuumcmd) {
            // Get the list of relations to analyze
            List* rels = vstmt->rels;

            if (rels != NIL) {
                // Specific relations were named
                ListCell* lc = nullptr;
                foreach (lc, rels) {
                    VacuumRelation* vrel = (VacuumRelation*)lfirst(lc);
                    if (vrel->relation != nullptr) {
                        Oid rel_oid = RangeVarGetRelid(vrel->relation, NoLock, true);
                        if (OidIsValid(rel_oid) && pg::table_storage::instance().table_exists(rel_oid)) {
                            Relation rel = RelationIdGetRelation(rel_oid);
                            if (RelationIsValid(rel)) {
                                if (pg::inject_deeplake_statistics(rel)) {
                                    elog(INFO,
                                         "Injected pre-computed DeepLake statistics for table '%s'",
                                         RelationGetRelationName(rel));
                                }
                                RelationClose(rel);
                            }
                        }
                    }
                }
            } else {
                // ANALYZE with no table specified - analyze all DeepLake tables
                // This is handled by PostgreSQL which will call ANALYZE on each table,
                // so each individual table will be processed by the specific-relation path above
                // when PostgreSQL recursively analyzes each table.
            }
        }
    }

    // Handle CREATE VIEW statement - check after view is created
    if (IsA(pstmt->utilityStmt, ViewStmt)) {
        ViewStmt* stmt = (ViewStmt*)pstmt->utilityStmt;
        const char* view_name = stmt->view->relname;
        const char* schema_name = stmt->view->schemaname != nullptr ? stmt->view->schemaname : "public";

        // Get the view OID that was just created
        Oid view_oid = RangeVarGetRelid(stmt->view, NoLock, true);
        if (OidIsValid(view_oid)) {
            // Get the view's rewrite rules to access the underlying query
            Relation view_rel = RelationIdGetRelation(view_oid);
            if (RelationIsValid(view_rel)) {
                // Get the view's query rewrite rule
                if (view_rel->rd_rules != nullptr && view_rel->rd_rules->numLocks > 0) {
                    RewriteRule* rule = view_rel->rd_rules->rules[0];
                    if (rule != nullptr && list_length(rule->actions) > 0) {
                        Query* query = (Query*)linitial(rule->actions);

                        if (query != nullptr && IsA(query, Query)) {
                            bool all_tables_deeplake = true;

                            // Check all tables in the FROM clause
                            ListCell* lc = nullptr;
                            foreach (lc, query->rtable) {
                                RangeTblEntry* rte = (RangeTblEntry*)lfirst(lc);
                                if (rte->rtekind == RTE_RELATION) {
                                    Oid rel_oid = rte->relid;
                                    if (!pg::table_storage::instance().table_exists(rel_oid)) {
                                        all_tables_deeplake = false;
                                        break;
                                    }
                                }
                            }

                            // If all tables use deeplake access method, print the view info
                            if (all_tables_deeplake && query->rtable != NIL) {
                                pg::table_storage::instance().add_view(view_oid, view_name, queryString);
                            }
                        }
                    }
                }
                RelationClose(view_rel);
            }
        }
    }
    if (IsA(pstmt->utilityStmt, VariableSetStmt)) {
        VariableSetStmt* vstmt = (VariableSetStmt*)pstmt->utilityStmt;
        if (vstmt->name != nullptr && pg_strcasecmp(vstmt->name, "search_path") == 0) {
            std::string schema_name = "public";
            if (vstmt->args != NIL && list_length(vstmt->args) == 1) {
                // Take only the first argument. deeplake executor supports one
                Node* arg = (Node*)linitial(vstmt->args);
                if (IsA(arg, A_Const)) {
                    A_Const* con = (A_Const*)arg;
                    if (IsA(&con->val, String)) {
                        schema_name = strVal(&con->val);
                    }
                }
            }
            pg::table_storage::instance().set_schema_name(std::move(schema_name));
        }
        // When root_path is set, auto-discover tables from the deeplake catalog
        if (vstmt->name != nullptr && pg_strcasecmp(vstmt->name, "deeplake.root_path") == 0) {
            // Reload table metadata from the catalog at the new root_path
            // This enables stateless multi-instance support where tables are
            // auto-discovered when pointing to a shared root_path
            pg::table_storage::instance().force_load_table_metadata();
        }
    }
}

// Check if the query string represents a pure SELECT statement (not CTAS, INSERT, etc.)
static bool is_pure_select_statement(const char* query_string)
{
    if (query_string == nullptr) {
        return false;
    }

    // Use PG_TRY to catch parse errors from invalid SQL (e.g., expression fragments
    // from plpgsql functions during constant folding)
    bool result = false;
    PG_TRY();
    {
        List* raw_parsetree_list = raw_parser(query_string, RAW_PARSE_DEFAULT);
        if (raw_parsetree_list != NIL) {
            RawStmt* raw_stmt = linitial_node(RawStmt, raw_parsetree_list);
            result = nodeTag(raw_stmt->stmt) == T_SelectStmt;
        }
    }
    PG_CATCH();
    {
        // Parse failed - not a valid SQL statement (e.g., expression fragment)
        FlushErrorState();
        result = false;
    }
    PG_END_TRY();

    return result;
}

static PlannedStmt*
deeplake_planner(Query* parse, const char* query_string, int32_t cursorOptions, ParamListInfo boundParams)
{
    pg::init_deeplake();

    // Transform ->> to jsonb_field_eq
    if (pg::support_json_index && parse && parse->jointree && parse->jointree->quals) {
        transform_jsonb_arrow_quals((Node**)&parse->jointree->quals);
    }

    PlannedStmt* planned_stmt = nullptr;
    if (pg::use_deeplake_executor && is_pure_select_statement(query_string)) {
        planned_stmt = deeplake_create_direct_execution_plan(parse, query_string, cursorOptions, boundParams);
    }

    if (!planned_stmt) {
        if (prev_planner_hook) {
            planned_stmt = prev_planner_hook(parse, query_string, cursorOptions, boundParams);
        } else {
            planned_stmt = standard_planner(parse, query_string, cursorOptions, boundParams);
        }
    }

    return planned_stmt;
}

static void executor_start(QueryDesc* query_desc, int32_t eflags)
{
    pg::init_deeplake();
    if (prev_executor_start != nullptr) {
        prev_executor_start(query_desc, eflags);
    } else {
        standard_ExecutorStart(query_desc, eflags);
    }

    if (!pg::query_info::is_in_top_context() || pg::utils::spi_connector::is_execution_in_progress()) {
        return;
    }

    pg::query_info::push_context(query_desc);
    pg::query_info::current().set_command_type(static_cast<enum pg::command_type>(query_desc->operation));

    if (!pg::query_info::current().is_deeplake_table_referenced()) {
        pg::analyze_plan(query_desc->plannedstmt);
    }
    if (query_desc->operation == CMD_SELECT && !pg::query_info::current().is_deeplake_table_referenced()) {
        return;
    }

    Plan* plan = query_desc->plannedstmt->planTree;
    if (plan != nullptr && IsA(plan, Limit)) {
        Limit* limitNode = (Limit*)plan;
        if (limitNode->limitCount != nullptr) {
            const auto limit = DatumGetInt32(((Const*)limitNode->limitCount)->constvalue);
            pg::query_info::current().set_limit(limit);
        }
        plan = plan->lefttree;
    }

    if (plan != nullptr) {
        List* targetList = plan->targetlist;
        ListCell* lc = nullptr;
        foreach (lc, targetList) {
            TargetEntry* tle = (TargetEntry*)lfirst(lc);
            if (::is_count_star(tle)) {
                pg::query_info::current().set_count_star(true);
                break;
            }
        }
    }
}

struct ReceiverState
{
    DestReceiver pub;
    DestReceiver* prev_receiver = nullptr; // Pointer to the original receiver
    bool startup_called = false;           // Track if startup has been called
};

static bool receive_slot(TupleTableSlot* slot, DestReceiver* self)
{
    slot_getallattrs(slot);
    for (auto i = 0; i < slot->tts_nvalid; ++i) {
        FormData_pg_attribute* attr = TupleDescAttr(slot->tts_tupleDescriptor, i);
        // Skip dropped columns
        if (attr->attisdropped) {
            continue;
        }
        if (!slot->tts_isnull[i]) {
            auto res = pg::utils::parse_special_datum(slot->tts_values[i], pg::utils::get_base_type(attr->atttypid));
            if (!res.is_valid) {
                continue;
            }
            if (pg::table_storage::instance().table_exists(res.table_id)) {
                pg::table_scan t_scan(res.table_id, false, false);
                t_scan.set_current_position(res.row_id);
                t_scan.fetch_column(res.column_id, slot->tts_values[i], slot->tts_isnull[i]);
            }
        }
    }
    auto* state = (ReceiverState*)self;
    if (state && state->prev_receiver && state->prev_receiver->receiveSlot) {
        return state->prev_receiver->receiveSlot(slot, state->prev_receiver);
    }
    return true;
}

static void receive_startup(DestReceiver* self, int32_t operation, TupleDesc typeinfo)
{
    auto* state = (ReceiverState*)self;
    state->startup_called = true;

    if (state->prev_receiver && state->prev_receiver->rStartup) {
        state->prev_receiver->rStartup(state->prev_receiver, operation, typeinfo);
    }
}

static void receive_shutdown(DestReceiver* self)
{
    auto* state = (ReceiverState*)self;
    if (state->startup_called && state->prev_receiver && state->prev_receiver->rShutdown) {
        state->prev_receiver->rShutdown(state->prev_receiver);
    }
}

static void receive_destroy(DestReceiver* self)
{
    auto* state = (ReceiverState*)self;
    if (state->prev_receiver && state->prev_receiver->rDestroy) {
        state->prev_receiver->rDestroy(state->prev_receiver);
    }
    // Don't free here - we'll handle it in executor_run
}

static void executor_run(QueryDesc* query_desc,
                         ScanDirection direction,
                         uint64 count
#if PG_VERSION_NUM < PG_VERSION_NUM_18
                         ,
                         bool execute_once
#endif
)
{
    ReceiverState* receiver = nullptr;
    DestReceiver* orig = query_desc->dest;

    if (pg::query_info::current().has_deferred_fetch()) {
        receiver = (ReceiverState*)palloc0(sizeof(ReceiverState));
        receiver->prev_receiver = orig;
        receiver->startup_called = false;

        // Set up the wrapper receiver
        receiver->pub.receiveSlot = receive_slot;
        receiver->pub.rStartup = receive_startup;
        receiver->pub.rShutdown = receive_shutdown;
        receiver->pub.rDestroy = receive_destroy;
        receiver->pub.mydest = orig->mydest;

        query_desc->dest = &receiver->pub;
    }

    PG_TRY();
    {
        if (prev_executor_run != nullptr) {
            prev_executor_run(query_desc,
                              direction,
                              count
#if PG_VERSION_NUM < PG_VERSION_NUM_18
                              ,
                              execute_once
#endif
            );
        } else {
            standard_ExecutorRun(query_desc,
                                 direction,
                                 count
#if PG_VERSION_NUM < PG_VERSION_NUM_18
                                 ,
                                 execute_once
#endif
            );
        }
        if (pg::query_info::current().has_deferred_fetch()) {
            query_desc->dest = orig;
        }
    }
    PG_CATCH();
    {
        if (pg::query_info::current().has_deferred_fetch()) {
            query_desc->dest = orig;
        }
        try {
            pg::table_storage::instance().rollback_all();
            pg::query_info::cleanup();
            pg::table_storage::instance().reset_requested_columns();
        } catch (const std::exception& e) {
            // Silently handle cleanup errors
        } catch (...) {
            // Silently handle cleanup errors
        }
        PG_RE_THROW();
    }
    PG_END_TRY();
}

static void executor_end(QueryDesc* query_desc)
{
    if (prev_executor_end != nullptr) {
        prev_executor_end(query_desc);
    } else {
        standard_ExecutorEnd(query_desc);
    }

    if (pg::query_info::is_in_executor_context(query_desc)) {
        if (query_desc->operation == CMD_INSERT || query_desc->operation == CMD_UPDATE ||
            query_desc->operation == CMD_DELETE || query_desc->operation == CMD_UTILITY) {
            // Use PG_TRY/CATCH to handle errors during flush without cascading aborts
            PG_TRY();
            {
                if (!pg::table_storage::instance().flush_all()) {
                    pg::table_storage::instance().rollback_all();
                    ereport(ERROR, (errcode(ERRCODE_INTERNAL_ERROR), errmsg("Failed to flush table storage")));
                }
            }
            PG_CATCH();
            {
                // Error occurred during flush - rollback and suppress to prevent cascade
                // This prevents "Deeplake does not support transaction aborts" cascade
                pg::table_storage::instance().rollback_all();
                // Don't re-throw - let the transaction abort naturally
                FlushErrorState();
            }
            PG_END_TRY();
        }
        pg::query_info::pop_context(query_desc);
        pg::table_storage::instance().reset_requested_columns();
    }
}

static void set_rel_pathlist(PlannerInfo* root, RelOptInfo* rel, Index rti, RangeTblEntry* rte)
{
    if (prev_set_rel_pathlist_hook) {
        prev_set_rel_pathlist_hook(root, rel, rti, rte);
    }
    if (rte->relkind == RELKIND_RELATION && pg::table_storage::instance().table_exists(rte->relid)) {
        if (pg::use_parallel_workers) {
            rel->consider_parallel = true;
            rel->rel_parallel_workers = max_parallel_workers;
        } else {
            rel->consider_parallel = false;
            rel->rel_parallel_workers = 0;
            rel->partial_pathlist = NIL;
            rel->serverid = InvalidOid;
        }
    }
}

/// @}

/// @name: Exported functions
/// @description: These functions are exported to PostgreSQL
/// @{

PGDLLEXPORT Datum create_deeplake_table(PG_FUNCTION_ARGS)
{
    const std::string table_name(text_to_cstring(PG_GETARG_TEXT_P(0)));
    const std::string dataset_path(text_to_cstring(PG_GETARG_TEXT_P(1)));

    std::shared_ptr<deeplake_api::read_only_dataset> ds;
    try {
        ds = deeplake_api::open_read_only(dataset_path, {}).get_future().get();
    } catch (const base::exception& e) {
        ereport(ERROR, (errcode(ERRCODE_INTERNAL_ERROR), errmsg("%s", e.what())));
    }
    if (!ds) {
        ereport(ERROR,
                (errcode(ERRCODE_INVALID_PARAMETER_VALUE), errmsg("Dataset does not exist: %s", dataset_path.c_str())));
    }

    // Parse schema.table format and quote identifiers
    std::string schema_name = "public";
    std::string simple_table_name = table_name;
    size_t dot_pos = table_name.find('.');
    if (dot_pos != std::string::npos) {
        schema_name = table_name.substr(0, dot_pos);
        simple_table_name = table_name.substr(dot_pos + 1);
    }

    std::ostringstream sql;
    sql << "CREATE TABLE " << quote_identifier(schema_name.c_str()) << "." << quote_identifier(simple_table_name.c_str()) << " (";

    bool first = true;
    for (const auto& column : *ds) {
        if (!first) {
            sql << ", ";
        }
        first = false;
        const auto pg_type_str = pg::utils::pg_try([t = column.type()]() {
            return pg::utils::nd_to_pg_type(t);
        });
        sql << quote_identifier(column.name().c_str()) << " " << pg_type_str;
    }

    sql << ") USING deeplake WITH (dataset_path=" << quote_literal_cstr(dataset_path.c_str()) << ");";

    elog(INFO, "Execute SQL command to create table:\n%s", sql.str().c_str());

    pg::utils::spi_connector connector;
    if (SPI_execute(sql.str().c_str(), false, 0) != SPI_OK_UTILITY) {
        ereport(ERROR, (errcode(ERRCODE_INTERNAL_ERROR), errmsg("Failed to create table: %s", sql.str().c_str())));
    }

    PG_RETURN_VOID();
}

PGDLLEXPORT Datum deeplake_tableam_handler(PG_FUNCTION_ARGS)
{
    PG_RETURN_POINTER(&pg::deeplake_table_am_routine::routine);
}

/// @}

/// This will be called when when extension is loaded
PGDLLEXPORT void _PG_init()
{
    if (!process_shared_preload_libraries_in_progress) {
        ereport(ERROR,
                (errmsg("pg_deeplake should be loaded with shared_preload_libraries"),
                 errhint("Add pg_deeplake to shared_preload_libraries.")));
    }

    // Set up shared memory request hook (must be first)
    prev_shmem_request_hook = shmem_request_hook;
    shmem_request_hook = deeplake_shmem_request;

    // Set up shared memory startup hook
    prev_shmem_startup_hook = shmem_startup_hook;
    shmem_startup_hook = deeplake_shmem_startup;

    register_deeplake_executor();

    prev_executor_start = ExecutorStart_hook;
    ExecutorStart_hook = executor_start;

    prev_executor_run = ExecutorRun_hook;
    ExecutorRun_hook = executor_run;

    prev_executor_end = ExecutorEnd_hook;
    ExecutorEnd_hook = executor_end;

    prev_process_utility_hook = ProcessUtility_hook;
    ProcessUtility_hook = process_utility;

    prev_planner_hook = planner_hook;
    planner_hook = deeplake_planner;

    prev_set_rel_pathlist_hook = set_rel_pathlist_hook;
    set_rel_pathlist_hook = set_rel_pathlist;

    // Initialize GUC parameters first (needed for sync worker config)
    ::initialize_guc_parameters();

    // Register background sync worker for stateless multi-instance support
    BackgroundWorker worker;
    memset(&worker, 0, sizeof(worker));

    snprintf(worker.bgw_name, BGW_MAXLEN, "pg_deeplake sync worker");
    snprintf(worker.bgw_type, BGW_MAXLEN, "pg_deeplake sync worker");
    snprintf(worker.bgw_library_name, BGW_MAXLEN, "pg_deeplake");
    snprintf(worker.bgw_function_name, BGW_MAXLEN, "deeplake_sync_worker_main");

    worker.bgw_flags = BGWORKER_SHMEM_ACCESS | BGWORKER_BACKEND_DATABASE_CONNECTION;
    worker.bgw_start_time = BgWorkerStart_RecoveryFinished;
    worker.bgw_restart_time = 5;  // Restart after 5 seconds if it crashes
    worker.bgw_notify_pid = 0;
    worker.bgw_main_arg = (Datum)0;

    RegisterBackgroundWorker(&worker);

    pg::install_signal_handlers();
    pg::deeplake_table_am_routine::initialize();
}

PGDLLEXPORT void _PG_fini()
{
    ExecutorStart_hook = prev_executor_start;
    ExecutorRun_hook = prev_executor_run;
    ExecutorEnd_hook = prev_executor_end;
    ProcessUtility_hook = prev_process_utility_hook;

    std::promise<void> promise;
    async::submit_in_main([&promise]() {
        promise.set_value();
    });
    promise.get_future().get();
    storage::storage::deinitialize();
}

#ifdef __cplusplus
} /// extern "C"
#endif
