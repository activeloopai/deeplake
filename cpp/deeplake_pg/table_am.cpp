#ifdef __cplusplus
extern "C" {
#endif

// Must be first to avoid macro conflicts
#include <postgres.h>

#include <access/parallel.h>
#include <access/reloptions.h>  // For relation options
#include <access/tableam.h>
#include <access/xact.h>
#include <catalog/namespace.h>
#include <catalog/storage.h>
#include <miscadmin.h>
#include <nodes/bitmapset.h>  // For bitmap operations
#include <nodes/nodes.h>     // For node types
#include <nodes/parsenodes.h> // For parse nodes
#include <storage/block.h>
#include <storage/relfilelocator.h>
#include <utils/builtins.h>  // For text conversion functions
#include <utils/lsyscache.h>
#include <utils/rel.h>
#include <utils/relcache.h>
#include <utils/varlena.h>   // For text functions

#ifdef __cplusplus
}
#endif

#include "table_am.hpp"
#include "exceptions.hpp"
#include "logger.hpp"
#include "memory_tracker.hpp"
#include "pg_deeplake.hpp"
#include "pg_version_compat.h"
#include "progress_utils.hpp"
#include "table_scan.hpp"
#include "table_storage.hpp"
#include "table_version.hpp"

#include <cstddef>

namespace {

static constexpr int64_t num_tuples_to_reset_memory_context = 10000;

// Extended TableScanDesc with embedded Deeplake scan data
struct DeeplakeScanData
{
    TableScanDescData postgres_scan; // keep this first
    pg::table_scan scan_state;
    MemoryContext memory_context = nullptr;
    
    // TID range scanning state
    bool tid_range_scan_active = false;
    int64_t min_tid = 0;
    int64_t max_tid = 0;
    int64_t current_tid = 0;
    
    // Bitmap scanning state
    bool bitmap_scan_active = false;
    struct TBMIterateResult* current_tbmres = nullptr;
    int32_t current_offset = InvalidOffsetNumber;
    std::vector<int64_t> bitmap_row_numbers;
    
    // Sample scanning state
    bool sample_scan_active = false;
    struct SampleScanState* current_sample_scanstate = nullptr;

    bool print_progress = false;
    pg::utils::progress_display progress_bar;
    int64_t num_rescans = 0;

    void reset()
    {
        tid_range_scan_active = false;
        min_tid = 0;
        max_tid = 0;
        current_tid = 0;
        bitmap_scan_active = false;
        bitmap_row_numbers.clear();
        current_tbmres = nullptr;
        current_offset = InvalidOffsetNumber;
        sample_scan_active = false;
        current_sample_scanstate = nullptr;
    }

    DeeplakeScanData(pg::table_scan&& state)
        : scan_state(std::move(state))
    {
        memory_context = AllocSetContextCreate(CurrentMemoryContext, "scan_context", ALLOCSET_SMALL_SIZES);
        const auto num_rows = scan_state.get_table_data().num_rows();
        print_progress = pg::print_progress_during_seq_scan && num_rows > 100000;
        if (print_progress) {
            progress_bar = pg::utils::progress_display(num_rows,
                                                       "Progress of sequential scan for table '" +
                                                       scan_state.get_table_data().get_table_name() + "' " +
                                                       "(" + std::to_string(num_rows) + " rows)");
        }
    }
    DeeplakeScanData(const DeeplakeScanData&) = delete;
    DeeplakeScanData& operator=(const DeeplakeScanData&) = delete;
    ~DeeplakeScanData()
    {
        if (memory_context != nullptr) {
            MemoryContextDelete(memory_context);
        }
    }
};

inline DeeplakeScanData* get_scan_data(TableScanDesc scan)
{
    return reinterpret_cast<DeeplakeScanData*>(scan);
}

struct DeeplakeIndexFetchData
{
    IndexFetchTableData base;
    pg::table_scan scan_state;
    MemoryContext memory_context = nullptr;

    DeeplakeIndexFetchData(pg::table_scan&& state)
        : scan_state(std::move(state))
    {
        memory_context = AllocSetContextCreate(CurrentMemoryContext, "scan_context", ALLOCSET_SMALL_SIZES);
    }

    ~DeeplakeIndexFetchData()
    {
        if (memory_context != nullptr) {
            MemoryContextDelete(memory_context);
        }
    }

    void reset()
    {
        scan_state.reset_scan();
        MemoryContextReset(memory_context);
    }
};

inline DeeplakeIndexFetchData* get_index_fetch_data(IndexFetchTableData* data)
{
    return reinterpret_cast<DeeplakeIndexFetchData*>(data);
}

std::string get_qualified_table_name(Relation rel)
{
    Oid nspid = RelationGetNamespace(rel);
    char* nspname = get_namespace_name(nspid);
    std::string qualified_name = std::string(nspname ? nspname : "public") + "." + RelationGetRelationName(rel);
    if (nspname) {
        pfree(nspname);
    }
    return qualified_name;
}

bool deeplake_relation_needs_toast_table(Relation)
{
    return false;
}

uint64_t deeplake_relation_size(Relation rel, ForkNumber)
{
    auto table_id = RelationGetRelid(rel);
    if (pg::table_storage::instance().table_exists(table_id)) {
        auto& table_data = pg::table_storage::instance().get_table_data(table_id);
        auto total_bytes = heimdall::dataset_total_bytes(*table_data.get_read_only_dataset());
        return total_bytes / BLCKSZ;
    }
    return BLCKSZ; // Default block size in bytes
}

void deeplake_index_validate_scan(Relation heap_rel,
                                  Relation index_rel,
                                  struct IndexInfo* index_info,
                                  Snapshot snapshot,
                                  struct ValidateIndexState* state)
{
    // No-op for now, we'll need to implement proper index validation later
}

void deeplake_estimate_rel_size(Relation rel,
                                int32_t* attr_widths,
                                BlockNumber* pages,
                                double* tuples,
                                double* allvisfrac)
{
    auto table_id = RelationGetRelid(rel);
    if (pg::table_storage::instance().table_exists(table_id)) {
        auto& table_data = pg::table_storage::instance().get_table_data(table_id);
        if (tuples != nullptr) {
            *tuples = table_data.num_rows();
        }
        if (allvisfrac != nullptr) {
            *allvisfrac = 1.0; // Assume all tuples are visible
        }
        auto avg_row_width = 0;
        if (attr_widths != nullptr) {
            for (int32_t i = 0; i < table_data.num_columns(); ++i) {
                attr_widths[i] = pg::utils::get_column_width(table_data.get_base_atttypid(i), table_data.get_atttypmod(i));
                attr_widths[i] = std::max(8, attr_widths[i]);
                avg_row_width += attr_widths[i];
            }
            avg_row_width /= table_data.num_columns();
        }
        if (pages != nullptr) {
            constexpr uint32_t min_pages = 1;
            const uint32_t num_blocks = std::ceil(table_data.num_rows() / 65536.0);
            *pages = std::max(min_pages, num_blocks);
        }

        return;
    }
    // Conservative estimates
    if (pages != nullptr) {
        *pages = 10;
    }
    if (tuples != nullptr) {
        *tuples = 1000; // Assume small table by default
    }
    if (allvisfrac != nullptr) {
        *allvisfrac = 0.9;
    }
}

bool deeplake_scan_analyze_next_tuple(TableScanDesc scan,
                                      TransactionId OldestXmin,
                                      double* liverows,
                                      double* deadrows,
                                      TupleTableSlot* slot)
{
    CHECK_FOR_INTERRUPTS();

    DeeplakeScanData* scan_data = get_scan_data(scan);
    if (scan_data == nullptr) {
        return false;
    }

    // Try to fetch next tuple from your columnar storage
    if (!scan_data->scan_state.get_next_tuple(slot)) {
        return false;  // no more tuples
    }

    /*
     * For columnar DeepLake we don’t track tuple visibility like heap.
     * So treat everything as live.
     */
    if (liverows) {
        *liverows += 1.0;
    }

    return true;
}

#if PG_VERSION_NUM >= PG_VERSION_NUM_17
bool deeplake_scan_analyze_next_block(TableScanDesc scan, ReadStream* stream)
{
    return false;
}
#endif

double deeplake_index_build_range_scan(Relation heap_rel,
                                       Relation index_rel,
                                       struct IndexInfo* index_info,
                                       bool allow_sync,
                                       bool anyvisible,
                                       bool progress,
                                       BlockNumber start_blockno,
                                       BlockNumber numblocks,
                                       IndexBuildCallback callback,
                                       void* callback_state,
                                       TableScanDesc scan)
{
    const int32_t nkeys = index_info->ii_NumIndexKeyAttrs;
    AttrNumber* indexkeys = index_info->ii_IndexAttrNumbers;
    const auto table_id = RelationGetRelid(heap_rel);
    auto& td = pg::table_storage::instance().get_table_data(table_id);
    for (int32_t i = 0; i < nkeys; ++i) {
        int32_t attnum = indexkeys[i] - 1;
        if (attnum >= 0 && !td.column_has_streamer(attnum) && td.can_stream_column(attnum)) {
            td.create_streamer(attnum, -1);
        }
    }
    std::vector<Datum> values(nkeys, 0);
    std::vector<uint8_t> nulls(nkeys, 0);
    pg::table_scan tscan(table_id, false, false);
    const auto num_rows = td.num_rows();
    ItemPointerData tid;
    for (auto row = 0; row < num_rows; ++row) {
        auto [block_number, offset_number] = pg::utils::row_number_to_tid(row);
        ItemPointerSet(&tid, block_number, offset_number);
        for (int32_t i = 0; i < nkeys; ++i) {
            int32_t attnum = indexkeys[i] - 1;
            if (attnum < 0) [[unlikely]] {
                nulls[i] = true;
                values[i] = 0;
            } else [[likely]] {
                auto [value, null] = tscan.get_datum(attnum, row);
                values[i] = value;
                nulls[i] = null;
            }
        }
        callback(index_rel, &tid, values.data(), reinterpret_cast<bool*>(nulls.data()), true, callback_state);
    }

    return static_cast<double>(num_rows);
}

struct DeeplakeParallelScanDesc
{
    ParallelTableScanDescData base;

    bool is_initialized = false;
    void reset()
    {
        is_initialized = false;
    }
};

Size parallelscan_initialize(Relation rel, ParallelTableScanDesc pscan)
{
    if (!pg::use_parallel_workers) {
        return 0;
    }
    return sizeof(DeeplakeParallelScanDesc);
}

Size parallelscan_estimate(Relation rel)
{
    if (!pg::use_parallel_workers) {
        return 0;
    }
    return sizeof(DeeplakeParallelScanDesc);
}

void parallelscan_reinitialize(Relation rel, ParallelTableScanDesc pscan)
{
    // Reinitialize parallel scan state
    auto custom_pscan = (DeeplakeParallelScanDesc*)pscan;
    custom_pscan->reset();
}

TM_Result tuple_lock(Relation relation, ItemPointer tid, Snapshot snapshot,
                     TupleTableSlot* slot, CommandId cid, LockTupleMode mode,
                     LockWaitPolicy wait_policy, uint8_t flags, TM_FailureData* tmfd)
{
    return TM_Ok;  // Just allow the lock for now
}

bool tuple_fetch_row_version(Relation relation,
                             ItemPointer tid,
                             Snapshot snapshot,
                             TupleTableSlot* slot)
{
    CHECK_FOR_INTERRUPTS();

    // Switch to the slot's memory context
    pg::utils::memory_context_switcher context_switcher(slot->tts_mcxt);

    // Fetch the tuple using the existing table storage mechanism
    if (!pg::table_storage::instance().fetch_tuple(RelationGetRelid(relation), tid, slot)) {
        return false;
    }

    ExecStoreVirtualTuple(slot);
    return true;
}

bool tuple_tid_valid(TableScanDesc scan, ItemPointer tid)
{
    OffsetNumber off = ItemPointerGetOffsetNumber(tid);
    return off > 0 && off <= pg::DEEPLAKE_TUPLES_PER_BLOCK;
}

bool tuple_satisfies_snapshot(Relation relation,
                              TupleTableSlot* slot,
                              Snapshot snapshot)
{
    // Check if tuple satisfies snapshot
    return true;
}

void tuple_get_latest_tid(TableScanDesc scan, ItemPointer tid)
{
    auto table_id = RelationGetRelid(scan->rs_rd);
    auto& td = pg::table_storage::instance().get_table_data(table_id);
    auto [block_number, offset_number] = pg::utils::row_number_to_tid(td.num_total_rows());
    ItemPointerSet(tid, block_number, offset_number);
}

void convert_schema(TupleDesc tupdesc)
{
    for (auto i = 0; i < tupdesc->natts; ++i) {
        Form_pg_attribute attr = TupleDescAttr(tupdesc, i);
        if (attr->atttypid == NUMERICOID && pg::treat_numeric_as_double) {
            auto typmod = attr->atttypmod;
            auto precision = -1;
            auto scale = -1;
            if (typmod >= VARHDRSZ) {
                precision = ((typmod - VARHDRSZ) >> 16) & 0xffff;
                scale = (typmod - VARHDRSZ) & 0xffff;
            }
            // Convert to FLOAT8
            attr->atttypid = FLOAT8OID;
            attr->attlen = sizeof(float8);      // 8 bytes
            attr->attbyval = true;              // pass by value
            attr->attalign = TYPALIGN_DOUBLE;   // 'd' - double alignment
            attr->atttypmod = -1;               // no type modifier for FLOAT8
            attr->attndims = 0;                 // not an array
            const char* column_name = NameStr(attr->attname);
            if (precision >= 0) {
                elog(WARNING, "Column '%s' converted from NUMERIC(%d,%d) to FLOAT8 - precision may be lost", column_name, precision, scale);
            } else {
                elog(WARNING, "Column '%s' converted from NUMERIC to FLOAT8 - precision may be lost", column_name);
            }
        } else if (attr->atttypid == CHAROID || attr->atttypid == BPCHAROID) {
            // Convert CHAR and BPCHAR to VARCHAR
            attr->atttypid = VARCHAROID;
            attr->attlen = -1;                  // variable length
            attr->attbyval = false;             // pass by reference
            attr->attalign = TYPALIGN_INT;      // 'i' - int4 alignment
            attr->attstorage = TYPSTORAGE_EXTENDED;
            const char* column_name = NameStr(attr->attname);
            elog(WARNING, "Column '%s' converted from CHAR/BPCHAR to VARCHAR as no fixed length string is supported", column_name);
        }
        Oid base_typeid = pg::utils::get_base_type(attr->atttypid);
        // For domain types over arrays, adjust attndims since PostgreSQL doesn't preserve it
        const bool is_domain_over_array = (base_typeid != attr->atttypid && type_is_array(base_typeid));
        if (is_domain_over_array && attr->attndims == 0) {
            // Extract dimensionality from domain's CHECK constraint
            // Query pg_constraint and use pg_get_constraintdef() to get the constraint text
            auto query_str = fmt::format("SELECT pg_get_constraintdef(oid) FROM pg_constraint WHERE contypid = {} AND contype = 'c'", attr->atttypid);
            pg::utils::spi_connector connector;
            bool found_constraint = false;
            if (SPI_execute(query_str.c_str(), true, 0) == SPI_OK_SELECT && SPI_processed > 0) {
                // Parse the constraint expression to find array_ndims
                for (uint64_t row = 0; row < SPI_processed; ++row) {
                    HeapTuple tuple = SPI_tuptable->vals[row];
                    bool isnull = false;
                    Datum condef_datum = SPI_getbinval(tuple, SPI_tuptable->tupdesc, 1, &isnull);
                    if (isnull) {
                        continue;
                    }
                    const char* condef = TextDatumGetCString(condef_datum);
                    // Look for patterns like "array_ndims(...) = N" or "CHECK (array_ndims(...) = N)"
                    const char* ndims_pos = strstr(condef, "array_ndims");
                    if (ndims_pos != nullptr) {
                        const char* eq_pos = strchr(ndims_pos, '=');
                        if (eq_pos != nullptr) {
                            eq_pos++;
                            while (*eq_pos == ' ' || *eq_pos == '\t') eq_pos++;
                            // Parse the integer (atoi handles leading whitespace and stops at non-digit)
                            char* endptr;
                            auto ndims = strtol(eq_pos, &endptr, 10);
                            // Verify we actually parsed a number and it's reasonable
                            if (endptr != eq_pos && ndims > 0 && ndims <= 2) {
                                attr->attndims = static_cast<int32_t>(ndims);
                                found_constraint = true;
                                break;
                            }
                        }
                    }
                }
            }
            // If no array_ndims constraint found, require user to add one
            if (!found_constraint) {
                const char* tname = format_type_with_typemod(attr->atttypid, attr->atttypmod);
                const char* column_name = NameStr(attr->attname);
                elog(ERROR,
                     "Column '%s': Domain type '%s' over array type lacks array_ndims constraint.\n"
                     "Please add a CHECK constraint, e.g.:\n"
                     "  ALTER DOMAIN %s ADD CHECK (array_ndims(VALUE) = 1);",
                     column_name, tname, tname);
            }
        }
    }
}

} // unnamed namespace

namespace pg {

TableAmRoutine deeplake_table_am_routine::routine;

void deeplake_table_am_routine::initialize()
{
    // Set the routine properties
    routine.type = T_TableAmRoutine;

    // Set up the slot callbacks
    routine.slot_callbacks = slot_callbacks;

    // Scan related callbacks
    routine.scan_begin = scan_begin;
    routine.scan_rescan = scan_rescan;
    routine.scan_end = scan_end;
    routine.scan_getnextslot = scan_getnextslot;
    
    // TID range scanning callbacks
    routine.scan_set_tidrange = scan_set_tidrange;
    routine.scan_getnextslot_tidrange = scan_getnextslot_tidrange;

    // Index scan related callbacks
    routine.index_fetch_begin = begin_index_fetch;
    routine.index_fetch_end = end_index_fetch;
    routine.index_fetch_tuple = index_fetch_tuple;
    routine.index_fetch_reset = index_fetch_reset;

    // Table operations
    routine.tuple_insert = tuple_insert;
    routine.multi_insert = multi_insert;
    routine.tuple_delete = tuple_delete;
    routine.tuple_update = tuple_update;

    routine.tuple_lock = tuple_lock;
    routine.tuple_fetch_row_version = tuple_fetch_row_version;
    routine.tuple_get_latest_tid = tuple_get_latest_tid;
    routine.tuple_tid_valid = tuple_tid_valid;
    routine.tuple_satisfies_snapshot = tuple_satisfies_snapshot;

    // Table creation and maintenance
    routine.relation_set_new_filelocator = relation_set_new_node;
    routine.relation_nontransactional_truncate = relation_nontransactional_truncate;

    // TOAST table support - disable TOAST tables completely
    routine.relation_needs_toast_table = deeplake_relation_needs_toast_table;

    // Index support
    routine.index_build_range_scan = deeplake_index_build_range_scan;
    routine.index_validate_scan = deeplake_index_validate_scan;

    // Optional scan methods
    ///routine.scan_bitmap_next_block = scan_bitmap_next_block;
#if PG_VERSION_NUM >= PG_VERSION_NUM_18
    routine.scan_bitmap_next_tuple = scan_bitmap_next_tuple;
#endif

    routine.relation_size = deeplake_relation_size;
    routine.relation_estimate_size = deeplake_estimate_rel_size;
    routine.scan_analyze_next_tuple = deeplake_scan_analyze_next_tuple;
#if PG_VERSION_NUM >= PG_VERSION_NUM_17
    routine.scan_analyze_next_block = deeplake_scan_analyze_next_block;
#endif

    routine.parallelscan_initialize = parallelscan_initialize;
    routine.parallelscan_estimate = parallelscan_estimate;
    routine.parallelscan_reinitialize = parallelscan_reinitialize;
}

const TupleTableSlotOps* deeplake_table_am_routine::slot_callbacks(Relation rel)
{
    // TODO: check if we use virtual table slot ops will be better
    return &TTSOpsHeapTuple;
}

TableScanDesc deeplake_table_am_routine::scan_begin(Relation relation,
                                                    Snapshot snapshot,
                                                    int32_t nkeys,
                                                    struct ScanKeyData* key,
                                                    ParallelTableScanDesc parallel_scan,
                                                    uint32_t flags)
{
    pg::runtime_printer scan_begin_timer("Table Scan Begin");
    DeeplakeScanData* extended_scan = (DeeplakeScanData*)palloc0(sizeof(DeeplakeScanData));
    auto table_id = RelationGetRelid(relation);
    bool is_parallel = (pg::use_parallel_workers && parallel_scan != nullptr);

    // Initialize extended structure with embedded scan data
    new (extended_scan) DeeplakeScanData(table_scan(table_id, is_parallel, query_info::current().receiver_registered()));
    const_cast<pg::table_data&>(extended_scan->scan_state.get_table_data()).refresh();

    TableScanDesc scan_desc = &extended_scan->postgres_scan;
    scan_desc->rs_rd = relation;
    scan_desc->rs_snapshot = snapshot;
    scan_desc->rs_nkeys = nkeys;
    scan_desc->rs_key = key;
    scan_desc->rs_flags = flags;
    scan_desc->rs_parallel = parallel_scan;

    if (parallel_scan != nullptr) {
        DeeplakeParallelScanDesc* custom_pscan = (DeeplakeParallelScanDesc*)parallel_scan;
        if (IsParallelWorker()) {
            custom_pscan->is_initialized = true;
        }
    }

    auto& td = table_storage::instance().get_table_data(table_id);
    if (!pg::query_info::current().is_count_star() && td.is_star_selected()) {
        for (auto i = 0; i < td.num_columns(); ++i) {
            if (!td.is_column_requested(i)) {
                td.set_column_requested(i, true);
                if (td.can_stream_column(i)) {
                    td.create_streamer(i, -1);
                }
            }
        }
    }

    if (nkeys > 0) {
        extended_scan->scan_state.nkeys = nkeys;
        // copy ScanKeyData because Postgres only gave us a pointer
        extended_scan->scan_state.keys = (ScanKeyData*) palloc(sizeof(ScanKeyData) * nkeys);
        std::memcpy(extended_scan->scan_state.keys, key, sizeof(ScanKeyData) * nkeys);
    }

    return scan_desc;
}

void deeplake_table_am_routine::scan_rescan(TableScanDesc scan, struct ScanKeyData* key,
                                            bool set_params, bool allow_strat,
                                            bool allow_sync, bool allow_pagemode)
{
    DeeplakeScanData* scan_data = get_scan_data(scan);
    if (scan_data) {
        scan_data->scan_state.set_current_position(0);
        MemoryContextReset(scan_data->memory_context);
        scan_data->reset();
        if (scan_data->print_progress) [[unlikely]] {
            ++scan_data->num_rescans;
            std::string pre_msg = "Progress of sequential scan for table '" +
                                  scan_data->scan_state.get_table_data().get_table_name() + "'" +
                                  " (rescan " + std::to_string(scan_data->num_rescans) + ")";
            scan_data->progress_bar.restart(scan_data->scan_state.get_table_data().num_rows(), std::move(pre_msg));
        }
    }
}

void deeplake_table_am_routine::scan_end(TableScanDesc scan)
{
    pg::runtime_printer scan_end_timer("Table Scan End");
    DeeplakeScanData* extended_scan = get_scan_data(scan);
    extended_scan->~DeeplakeScanData();
    pfree(extended_scan);
}

bool deeplake_table_am_routine::scan_getnextslot(TableScanDesc scan, ScanDirection direction, TupleTableSlot* slot)
{
    CHECK_FOR_INTERRUPTS();

    DeeplakeScanData* scan_data = get_scan_data(scan);

    // Reset memory context to prevent unbounded growth
    if (scan_data->scan_state.get_current_position() % num_tuples_to_reset_memory_context == 0) {
        MemoryContextReset(scan_data->memory_context);
    }

    // Switch to the dedicated memory context for this scan
    pg::utils::memory_context_switcher context_switcher(scan_data->memory_context);

    if (scan_data->print_progress) [[unlikely]] {
        ++scan_data->progress_bar;
    }

    return scan_data->scan_state.get_next_tuple(slot);
}

void deeplake_table_am_routine::scan_set_tidrange(TableScanDesc scan, ItemPointer mintid, ItemPointer maxtid)
{
    CHECK_FOR_INTERRUPTS();
    
    DeeplakeScanData* scan_data = get_scan_data(scan);
    scan_data->min_tid = utils::tid_to_row_number(mintid);
    scan_data->max_tid = utils::tid_to_row_number(maxtid);
    scan_data->current_tid = utils::tid_to_row_number(mintid);

    if (scan_data->min_tid > scan_data->max_tid ||
        scan_data->current_tid > scan_data->max_tid ||
        scan_data->current_tid < scan_data->min_tid) {
        return;
    }

    scan_data->tid_range_scan_active = true;

    // Reset the scan state to start from the beginning of the range
    scan_data->scan_state.set_current_position(scan_data->current_tid);
    MemoryContextReset(scan_data->memory_context);
}

bool deeplake_table_am_routine::scan_getnextslot_tidrange(TableScanDesc scan, ScanDirection direction, TupleTableSlot* slot)
{
    CHECK_FOR_INTERRUPTS();

    DeeplakeScanData* scan_data = get_scan_data(scan);
    if (!scan_data || !scan_data->tid_range_scan_active) {
        return false;
    }

    const auto current = scan_data->scan_state.get_current_position();
    if (current > scan_data->max_tid || current < scan_data->min_tid) {
        return false;
    }

    if (current % num_tuples_to_reset_memory_context == 0) {
        MemoryContextReset(scan_data->memory_context);
    }

    // Switch to the dedicated memory context for this scan
    pg::utils::memory_context_switcher context_switcher(scan_data->memory_context);

    return scan_data->scan_state.get_next_tuple(slot);
}

#if PG_VERSION_NUM >= PG_VERSION_NUM_18
bool deeplake_table_am_routine::scan_bitmap_next_tuple(TableScanDesc scan, TupleTableSlot* slot, bool* recheck, uint64* lossy_pages, uint64* exact_pages)
{
    CHECK_FOR_INTERRUPTS();
    DeeplakeScanData* scan_data = get_scan_data(scan);

    if (!scan_data->bitmap_scan_active) {
        scan_data->bitmap_scan_active = true;
        scan_data->current_offset = 0;
        *lossy_pages = 0;
        *exact_pages = 0;

        // Get the merged TBMIterator from scan descriptor
        TBMIterator* iter = &scan->st.rs_tbmiterator;
        TBMIterateResult tbmres;

        while (tbm_iterate(iter, &tbmres)) {
            if (tbmres.lossy) {
                // Lossy page - all tuples in this block
                int64_t block_start = static_cast<int64_t>(tbmres.blockno) * pg::DEEPLAKE_TUPLES_PER_BLOCK;
                int64_t block_end = block_start + pg::DEEPLAKE_TUPLES_PER_BLOCK;
                for (int64_t row = block_start; row < block_end; ++row) {
                    scan_data->bitmap_row_numbers.push_back(row);
                }
                ++(*lossy_pages);
            } else {
                // Exact tuples - extract offsets
                OffsetNumber offsets[TBM_MAX_TUPLES_PER_PAGE];
                int32_t ntuples = tbm_extract_page_tuple(&tbmres, offsets, TBM_MAX_TUPLES_PER_PAGE);
                for (int32_t i = 0; i < ntuples; i++) {
                    // Convert (block, offset) to row number
                    ItemPointerData tid;
                    ItemPointerSet(&tid, tbmres.blockno, offsets[i]);
                    int64_t row_num = pg::utils::tid_to_row_number(&tid);
                    scan_data->bitmap_row_numbers.push_back(row_num);
                }
                ++(*exact_pages);
            }
        }
    }

    if (scan_data->current_offset >= scan_data->bitmap_row_numbers.size()) {
        return false;  // No more tuples
    }

    *recheck = false;
    // Get next row number
    int64_t row_num = scan_data->bitmap_row_numbers[scan_data->current_offset++];
    scan_data->scan_state.set_current_position(row_num);
    return scan_data->scan_state.get_next_tuple(slot);
}
#endif

bool deeplake_table_am_routine::scan_sample_next_block(TableScanDesc scan, struct SampleScanState* scanstate)
{
    DeeplakeScanData* scan_data = get_scan_data(scan);
    if (!scan_data || scan_data->sample_scan_active) { // behaves as a single-block
        return false;
    }

    scan_data->current_sample_scanstate = scanstate;
    scan_data->sample_scan_active = true;

    return true;
}

bool deeplake_table_am_routine::scan_sample_next_tuple(TableScanDesc scan, struct SampleScanState* scanstate, TupleTableSlot* slot)
{
    CHECK_FOR_INTERRUPTS();

    DeeplakeScanData* scan_data = get_scan_data(scan);
    if (!scan_data || !scan_data->sample_scan_active) {
        return false;
    }

    // Reset memory context to prevent unbounded growth
    if (scan_data->scan_state.get_current_position() % num_tuples_to_reset_memory_context == 0) {
        MemoryContextReset(scan_data->memory_context);
    }

    // Switch to the dedicated memory context for this scan
    pg::utils::memory_context_switcher context_switcher(scan_data->memory_context);

    double sample_fraction = 0.1; // default fallback
    if (scan_data->current_sample_scanstate && scan_data->current_sample_scanstate->tsm_state) {
        // If your TAM stores the fraction in tsm_state, cast and read it here
        struct SampleState { double fraction; };
        SampleState* sampler = (SampleState*)scan_data->current_sample_scanstate->tsm_state;
        sample_fraction = sampler->fraction;
    }

    while (scan_data->scan_state.get_next_tuple(slot)) {
        // random fraction, should come from scanstate->tsm_state
        const double random_value = (double)random() / RAND_MAX;

        if (random_value <= sample_fraction) {
            return true;
        }
    }

    return false;
}

IndexFetchTableData* deeplake_table_am_routine::begin_index_fetch(Relation rel)
{
    DeeplakeIndexFetchData* idx_scan = (DeeplakeIndexFetchData*)palloc0(sizeof(DeeplakeIndexFetchData));
    new (idx_scan) DeeplakeIndexFetchData(table_scan(RelationGetRelid(rel), false, query_info::current().receiver_registered()));
    idx_scan->base.rel = rel;
    return &idx_scan->base;
}

void deeplake_table_am_routine::index_fetch_reset(IndexFetchTableData* data)
{
    DeeplakeIndexFetchData* idx_scan = get_index_fetch_data(data);
    idx_scan->reset();
}

void deeplake_table_am_routine::end_index_fetch(IndexFetchTableData* data)
{
    DeeplakeIndexFetchData* idx_scan = get_index_fetch_data(data);
    idx_scan->~DeeplakeIndexFetchData();
    pfree(idx_scan);
}

bool deeplake_table_am_routine::index_fetch_tuple(struct IndexFetchTableData* scan,
                                                  ItemPointer tid,
                                                  Snapshot snapshot,
                                                  TupleTableSlot* slot,
                                                  bool* call_again,
                                                  bool* all_dead)
{
    CHECK_FOR_INTERRUPTS();

    DeeplakeIndexFetchData* idx_scan = get_index_fetch_data(scan);

    if (idx_scan->scan_state.get_current_position() % num_tuples_to_reset_memory_context == 0) {
        MemoryContextReset(idx_scan->memory_context);
    }

    pg::utils::memory_context_switcher context_switcher(idx_scan->memory_context);
    idx_scan->scan_state.set_current_position(utils::tid_to_row_number(tid));
    if (!idx_scan->scan_state.get_next_tuple(slot)) {
        *all_dead = true;
        return false;
    }

    *call_again = false;
    *all_dead = false;
    return true;
}

void deeplake_table_am_routine::tuple_insert(Relation rel,
                                             TupleTableSlot* slot,
                                             CommandId cid,
                                             int32_t options,
                                             struct BulkInsertStateData* bistate)
{
    const auto table_id = RelationGetRelid(rel);
    const auto& table_data = table_storage::instance().get_table_data(table_id);
    const auto row_number = table_data.num_total_rows();
    table_storage::instance().insert_slot(table_id, slot);
    const auto [block_number, offset_number] = utils::row_number_to_tid(row_number);
    ItemPointerSet(&slot->tts_tid, block_number, offset_number);

    // Increment version to notify other backends
    table_version_tracker::increment_version(table_id);
}

void deeplake_table_am_routine::multi_insert(Relation rel,
                                             TupleTableSlot** slots,
                                             int32_t nslots,
								             CommandId cid,
                                             int32_t options,
                                             struct BulkInsertStateData* bistate)
{
    // Check PostgreSQL memory limits before multi-insert
    memory_tracker::check_memory_limit();
    const auto table_id = RelationGetRelid(rel);
    const auto& table_data = table_storage::instance().get_table_data(table_id);
    const auto row_number = table_data.num_total_rows();
    try {
        table_storage::instance().insert_slots(table_id, nslots, slots);
    } catch (const std::exception& e) {
        ereport(ERROR, (errcode(ERRCODE_INTERNAL_ERROR), errmsg("Failed to insert tuples: %s", e.what()),
                        errdetail("Table name: %s, Dataset path: %s",
                                  get_qualified_table_name(rel).c_str(),
                                  pg::table_options::current().dataset_path().c_str()),
                        errhint("Check if the dataset path is accessible and has proper permissions")));
    }
    for (auto i = 0; i < nslots; ++i) {
        const auto [block_number, offset_number] = utils::row_number_to_tid(row_number + i);
        ItemPointerSet(&slots[i]->tts_tid, block_number, offset_number);
    }

    // Increment version to notify other backends
    table_version_tracker::increment_version(table_id);
}

TM_Result deeplake_table_am_routine::tuple_delete(Relation rel,
                                                  ItemPointer tid,
                                                  CommandId cid,
                                                  Snapshot snapshot,
                                                  Snapshot crosscheck,
                                                  bool wait,
                                                  TM_FailureData* tmfd,
                                                  bool changingPart)
{
    const auto table_id = RelationGetRelid(rel);

    // Delete the tuple from our storage
    try {
        if (!table_storage::instance().delete_tuple(table_id, tid)) {
            ereport(ERROR, (errcode(ERRCODE_INTERNAL_ERROR), errmsg("Failed to delete tuple")));
        }
    } catch (const std::exception& e) {
        ereport(ERROR, (errcode(ERRCODE_INTERNAL_ERROR), errmsg("Failed to delete tuple: %s", e.what())));
    }

    // Increment version to notify other backends
    table_version_tracker::increment_version(table_id);

    return TM_Ok;
}

TM_Result deeplake_table_am_routine::tuple_update(Relation rel,
                                                  ItemPointer otid,
                                                  TupleTableSlot* slot,
                                                  CommandId cid,
                                                  Snapshot snapshot,
                                                  Snapshot crosscheck,
                                                  bool wait,
                                                  TM_FailureData* tmfd,
                                                  LockTupleMode* lockmode,
                                                  TU_UpdateIndexes* update_indexes)
{
    const auto table_id = RelationGetRelid(rel);

    // Convert slot to HeapTuple
    HeapTuple tuple = ExecFetchSlotHeapTuple(slot, true, nullptr);
    if (tuple == nullptr) {
        ereport(ERROR, (errcode(ERRCODE_INTERNAL_ERROR), errmsg("Failed to fetch tuple from slot")));
    }

    // Update the tuple in our storage
    try {
        if (!table_storage::instance().update_tuple(table_id, otid, tuple)) {
            ereport(ERROR, (errcode(ERRCODE_INTERNAL_ERROR), errmsg("Failed to update tuple")));
        }
    } catch (const std::exception& e) {
        ereport(ERROR, (errcode(ERRCODE_INTERNAL_ERROR), errmsg("Failed to update tuple: %s", e.what())));
    }

    // Increment version to notify other backends
    table_version_tracker::increment_version(table_id);

    return TM_Ok;
}

void deeplake_table_am_routine::relation_set_new_node(Relation rel,
                                                      const RelFileLocator* newrnode,
                                                      char persistence,
                                                      TransactionId* freezeXid,
                                                      MultiXactId* minmulti)
{
    // Get the schema-qualified table name
    const std::string table_name = get_qualified_table_name(rel);

    // Get the tuple descriptor
    TupleDesc tupdesc = RelationGetDescr(rel);
    if (tupdesc == nullptr) {
        ereport(ERROR,
                (errcode(ERRCODE_UNDEFINED_TABLE),
                 errmsg("could not get tuple descriptor for relation \"%s\"", RelationGetRelationName(rel))));
    }

    convert_schema(tupdesc);

    try {
        table_storage::instance().create_table(table_name, RelationGetRelid(rel), tupdesc);
    } catch (const std::exception& e) {
        ereport(ERROR,
                (errcode(ERRCODE_INTERNAL_ERROR),
                 errmsg("failed to create table storage: %s", e.what()),
                 errdetail("Table name: %s, Dataset path: %s",
                           table_name.c_str(),
                           pg::table_options::current().dataset_path().c_str()),
                 errhint("Check if the dataset path is accessible and has proper permissions")));
    }
}

void deeplake_table_am_routine::relation_nontransactional_truncate(Relation rel)
{
    return;
}

} // namespace pg
