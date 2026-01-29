#pragma once

// NOTE: postgres.h and libintl.h must be included first in the compilation unit before this header

#ifdef __cplusplus
extern "C" {
#endif

// PostgreSQL includes
#include <access/htup.h>
#include <access/htup_details.h>
#include <access/skey.h>
#include <access/table.h>
#include <access/tableam.h>
#include <access/tupdesc.h>
#include <storage/block.h>
#include <storage/relfilelocator.h>
#include <utils/rel.h>
#include <utils/snapmgr.h>

#ifdef __cplusplus
}
#endif

namespace pg {

struct deeplake_table_am_routine
{
    static TableAmRoutine routine; // The actual routine instance

    // Initialize and register the table access method
    static void initialize();

    // Core table operation callbacks
    static const TupleTableSlotOps* slot_callbacks(Relation rel);

    static TableScanDesc scan_begin(Relation relation,
                                    Snapshot snapshot,
                                    int32_t nkeys,
                                    struct ScanKeyData* key,
                                    ParallelTableScanDesc parallel_scan,
                                    uint32 flags);

    static void scan_rescan(TableScanDesc scan,
                            struct ScanKeyData* key,
                            bool set_params, bool allow_strat,
                            bool allow_sync, bool allow_pagemode);

    static void scan_end(TableScanDesc scan);

    static bool scan_getnextslot(TableScanDesc scan,
                                 ScanDirection direction,
                                 TupleTableSlot* slot);

    // TID range scanning functions
    static void scan_set_tidrange(TableScanDesc scan,
                                  ItemPointer mintid,
                                  ItemPointer maxtid);

    static bool scan_getnextslot_tidrange(TableScanDesc scan,
                                          ScanDirection direction,
                                          TupleTableSlot* slot);

    // Bitmap scanning functions
    static bool scan_bitmap_next_tuple(TableScanDesc scan,
                                       TupleTableSlot* slot,
                                       bool* recheck,
                                       uint64* lossy_pages,
                                       uint64* exact_pages);

    // Sample scanning functions
    static bool scan_sample_next_block(TableScanDesc scan,
                                       struct SampleScanState* scanstate);

    static bool scan_sample_next_tuple(TableScanDesc scan,
                                       struct SampleScanState* scanstate,
                                       TupleTableSlot* slot);

    static IndexFetchTableData* begin_index_fetch(Relation rel);
    static void index_fetch_reset(IndexFetchTableData* scan);
    static void end_index_fetch(IndexFetchTableData* data);

    static bool index_fetch_tuple(struct IndexFetchTableData* scan,
                                  ItemPointer tid,
                                  Snapshot snapshot,
                                  TupleTableSlot* slot,
                                  bool* call_again,
                                  bool* all_dead);

    static void tuple_insert(Relation rel,
                             TupleTableSlot* slot,
                             CommandId cid,
                             int32_t options,
                             struct BulkInsertStateData* bistate);

    static void multi_insert(Relation rel,
                             TupleTableSlot** slots,
                             int32_t nslots,
                             CommandId cid,
                             int32_t options,
                             struct BulkInsertStateData* bistate);

    static TM_Result tuple_delete(Relation rel,
                                  ItemPointer tid,
                                  CommandId cid,
                                  Snapshot snapshot,
                                  Snapshot crosscheck,
                                  bool wait,
                                  TM_FailureData* tmfd,
                                  bool changingPart);

    static TM_Result tuple_update(Relation rel,
                                  ItemPointer otid,
                                  TupleTableSlot* slot,
                                  CommandId cid,
                                  Snapshot snapshot,
                                  Snapshot crosscheck,
                                  bool wait,
                                  TM_FailureData* tmfd,
                                  LockTupleMode* lockmode,
                                  TU_UpdateIndexes* update_indexes);

    // Table creation and maintenance
    static void relation_set_new_node(Relation rel,
                                      const RelFileLocator* newrnode,
                                      char persistence,
                                      TransactionId* freezeXid,
                                      MultiXactId* minmulti);

    static void relation_nontransactional_truncate(Relation rel);
};

} // namespace pg
