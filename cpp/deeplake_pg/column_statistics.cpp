/**
 * @file column_statistics.cpp
 * @brief Injects pre-computed DeepLake column statistics into PostgreSQL's pg_statistic.
 *
 * This module provides integration between DeepLake's incremental statistics computation
 * and PostgreSQL's query planner. Instead of sampling data during ANALYZE, we use
 * statistics that were already computed during data ingestion.
 */

#include <libintl.h>

#ifdef __cplusplus
extern "C" {
#endif

#include <postgres.h>

#include <access/heapam.h>
#include <access/htup_details.h>
#include <access/relation.h>
#include <access/stratnum.h>
#include <access/table.h>
#include <catalog/indexing.h>
#include <catalog/pg_am_d.h>
#include <catalog/pg_statistic.h>
#include <catalog/pg_type.h>
#include <commands/defrem.h>
#include <parser/parse_oper.h>
#include <utils/array.h>
#include <utils/builtins.h>
#include <utils/lsyscache.h>
#include <utils/rel.h>
#include <utils/syscache.h>
#include <utils/typcache.h>

#ifdef __cplusplus
}
#endif

#include "column_statistics.hpp"
#include "pg_deeplake.hpp"
#include "utils.hpp"
#include "table_storage.hpp"

#include <deeplake_core/column_statistics.hpp>

namespace pg {

namespace {

/**
 * @brief Get the equality operator OID for a given type.
 */
Oid get_equality_operator(Oid typid)
{
    Oid opclass = InvalidOid;
    Oid eq_opr = InvalidOid;

    // Try to find an appropriate equality operator
    TypeCacheEntry* typcache = lookup_type_cache(typid, TYPECACHE_EQ_OPR);
    if (OidIsValid(typcache->eq_opr)) {
        return typcache->eq_opr;
    }

    // Fallback: try to get operator from btree opclass
    opclass = GetDefaultOpClass(typid, BTREE_AM_OID);
    if (OidIsValid(opclass)) {
        eq_opr = get_opfamily_member(get_opclass_family(opclass), typid, typid, BTEqualStrategyNumber);
    }

    return eq_opr;
}

/**
 * @brief Convert an MCV value (std::variant) to a PostgreSQL Datum based on the target type.
 */
Datum mcv_value_to_datum(const deeplake_core::mcv_value& val, Oid typid, bool* is_null)
{
    *is_null = false;

    return std::visit(
        [typid, is_null](auto&& arg) -> Datum {
            using T = std::decay_t<decltype(arg)>;

            if constexpr (std::is_same_v<T, int64_t>) {
                switch (typid) {
                case INT2OID:
                    return Int16GetDatum(static_cast<int16_t>(arg));
                case INT4OID:
                    return Int32GetDatum(static_cast<int32_t>(arg));
                case INT8OID:
                    return Int64GetDatum(arg);
                case FLOAT4OID:
                    return Float4GetDatum(static_cast<float>(arg));
                case FLOAT8OID:
                    return Float8GetDatum(static_cast<double>(arg));
                default:
                    *is_null = true;
                    return (Datum)0;
                }
            } else if constexpr (std::is_same_v<T, uint64_t>) {
                switch (typid) {
                case INT2OID:
                    return Int16GetDatum(static_cast<int16_t>(arg));
                case INT4OID:
                    return Int32GetDatum(static_cast<int32_t>(arg));
                case INT8OID:
                    return Int64GetDatum(static_cast<int64_t>(arg));
                case FLOAT4OID:
                    return Float4GetDatum(static_cast<float>(arg));
                case FLOAT8OID:
                    return Float8GetDatum(static_cast<double>(arg));
                default:
                    *is_null = true;
                    return (Datum)0;
                }
            } else if constexpr (std::is_same_v<T, double>) {
                switch (typid) {
                case INT2OID:
                    return Int16GetDatum(static_cast<int16_t>(arg));
                case INT4OID:
                    return Int32GetDatum(static_cast<int32_t>(arg));
                case INT8OID:
                    return Int64GetDatum(static_cast<int64_t>(arg));
                case FLOAT4OID:
                    return Float4GetDatum(static_cast<float>(arg));
                case FLOAT8OID:
                    return Float8GetDatum(arg);
                default:
                    *is_null = true;
                    return (Datum)0;
                }
            } else if constexpr (std::is_same_v<T, std::string>) {
                switch (typid) {
                case TEXTOID:
                case VARCHAROID:
                    return PointerGetDatum(cstring_to_text_with_len(arg.data(), arg.size()));
                default:
                    // Try string conversion for other types
                    return PointerGetDatum(cstring_to_text_with_len(arg.data(), arg.size()));
                }
            } else if constexpr (std::is_same_v<T, bool>) {
                if (typid == BOOLOID) {
                    return BoolGetDatum(arg);
                }
                *is_null = true;
                return (Datum)0;
            } else {
                *is_null = true;
                return (Datum)0;
            }
        },
        val);
}

/**
 * @brief Build MCV (Most Common Values) arrays from DeepLake statistics.
 *
 * @param stats The DeepLake column statistics
 * @param typid The PostgreSQL type OID of the column
 * @param[out] mcv_values Output array of MCV values
 * @param[out] mcv_freqs Output array of MCV frequencies
 * @param[out] num_mcv Number of MCVs
 * @return true if MCV data was successfully built
 */
bool build_mcv_arrays(const deeplake_core::column_statistics& stats,
                      Oid typid,
                      Datum** mcv_values,
                      float4** mcv_freqs,
                      int* num_mcv)
{
    if (!stats.has_mcv() || stats.most_common_vals.empty()) {
        *num_mcv = 0;
        return false;
    }

    const size_t n = stats.most_common_vals.size();
    *mcv_values = static_cast<Datum*>(palloc(n * sizeof(Datum)));
    *mcv_freqs = static_cast<float4*>(palloc(n * sizeof(float4)));

    size_t valid_count = 0;
    for (size_t i = 0; i < n; ++i) {
        bool is_null = false;
        Datum d = mcv_value_to_datum(stats.most_common_vals[i], typid, &is_null);
        if (!is_null) {
            (*mcv_values)[valid_count] = d;
            (*mcv_freqs)[valid_count] = static_cast<float4>(stats.most_common_freqs[i]);
            valid_count++;
        }
    }

    *num_mcv = static_cast<int>(valid_count);
    return valid_count > 0;
}

} // anonymous namespace

bool inject_column_statistics(Relation rel, int16_t attnum)
{
    const Oid relid = RelationGetRelid(rel);

    // Check if this is a DeepLake table
    if (!table_storage::instance().table_exists(relid)) {
        elog(DEBUG1, "inject_column_statistics: table %u not in table_storage", relid);
        return false;
    }

    const auto& table_data = table_storage::instance().get_table_data(relid);

    // Get attribute info
    TupleDesc tupdesc = RelationGetDescr(rel);
    if (attnum <= 0 || attnum > tupdesc->natts) {
        return false;
    }

    Form_pg_attribute attr = TupleDescAttr(tupdesc, attnum - 1);
    if (attr->attisdropped) {
        return false;
    }

    // Map PG attnum to logical column index (handles dropped columns correctly)
    const auto col_idx = table_data.logical_index_for_attnum(attnum);
    if (col_idx < 0) {
        return false;
    }

    heimdall::column_view_ptr column_view;
    try {
        column_view = table_data.get_column_view(col_idx);
    } catch (const std::exception& e) {
        elog(DEBUG1, "Failed to get column view for column %d: %s", attnum, e.what());
        return false;
    }

    if (!column_view) {
        return false;
    }

    // Get statistics from DeepLake
    deeplake_core::column_statistics dl_stats;
    try {
        dl_stats = column_view->statistics();
    } catch (const std::exception& e) {
        elog(DEBUG1, "Failed to get statistics for column %d: %s", attnum, e.what());
        return false;
    }

    if (!dl_stats.has_statistics) {
        elog(DEBUG1, "DeepLake statistics not available for column %s (attnum=%d)",
             NameStr(attr->attname), attnum);
        return false;
    }

    // Get type information
    Oid typid = attr->atttypid;
    int16_t typlen = attr->attlen;
    bool typbyval = attr->attbyval;
    char typalign = attr->attalign;
    Oid eq_opr = get_equality_operator(typid);
    Oid collation = attr->attcollation;

    // Prepare values for pg_statistic tuple
    Datum values[Natts_pg_statistic];
    bool nulls[Natts_pg_statistic];
    bool replaces[Natts_pg_statistic];

    for (int i = 0; i < Natts_pg_statistic; ++i) {
        nulls[i] = false;
        replaces[i] = true;
    }

    // Fixed columns
    values[Anum_pg_statistic_starelid - 1] = ObjectIdGetDatum(relid);
    values[Anum_pg_statistic_staattnum - 1] = Int16GetDatum(attnum);
    values[Anum_pg_statistic_stainherit - 1] = BoolGetDatum(false);

    // Core statistics from DeepLake
    values[Anum_pg_statistic_stanullfrac - 1] = Float4GetDatum(static_cast<float4>(dl_stats.null_frac));
    values[Anum_pg_statistic_stawidth - 1] = Int32GetDatum(dl_stats.avg_width);
    values[Anum_pg_statistic_stadistinct - 1] = Float4GetDatum(static_cast<float4>(dl_stats.n_distinct));

    // Initialize slot fields to empty
    for (int k = 0; k < STATISTIC_NUM_SLOTS; k++) {
        values[Anum_pg_statistic_stakind1 - 1 + k] = Int16GetDatum(0);
        values[Anum_pg_statistic_staop1 - 1 + k] = ObjectIdGetDatum(InvalidOid);
        values[Anum_pg_statistic_stacoll1 - 1 + k] = ObjectIdGetDatum(InvalidOid);
        nulls[Anum_pg_statistic_stanumbers1 - 1 + k] = true;
        values[Anum_pg_statistic_stanumbers1 - 1 + k] = (Datum)0;
        nulls[Anum_pg_statistic_stavalues1 - 1 + k] = true;
        values[Anum_pg_statistic_stavalues1 - 1 + k] = (Datum)0;
    }

    // Slot 1: MCV (Most Common Values) if available
    if (dl_stats.has_mcv() && OidIsValid(eq_opr)) {
        Datum* mcv_values = nullptr;
        float4* mcv_freqs = nullptr;
        int num_mcv = 0;

        if (build_mcv_arrays(dl_stats, typid, &mcv_values, &mcv_freqs, &num_mcv) && num_mcv > 0) {
            // Set stakind1 = STATISTIC_KIND_MCV (1)
            values[Anum_pg_statistic_stakind1 - 1] = Int16GetDatum(STATISTIC_KIND_MCV);
            values[Anum_pg_statistic_staop1 - 1] = ObjectIdGetDatum(eq_opr);
            values[Anum_pg_statistic_stacoll1 - 1] = ObjectIdGetDatum(collation);

            // Build frequency array
            Datum* freq_datums = static_cast<Datum*>(palloc(num_mcv * sizeof(Datum)));
            for (int i = 0; i < num_mcv; ++i) {
                freq_datums[i] = Float4GetDatum(mcv_freqs[i]);
            }
            ArrayType* freq_array = construct_array_builtin(freq_datums, num_mcv, FLOAT4OID);
            values[Anum_pg_statistic_stanumbers1 - 1] = PointerGetDatum(freq_array);
            nulls[Anum_pg_statistic_stanumbers1 - 1] = false;

            // Build values array
            ArrayType* val_array = construct_array(mcv_values, num_mcv, typid, typlen, typbyval, typalign);
            values[Anum_pg_statistic_stavalues1 - 1] = PointerGetDatum(val_array);
            nulls[Anum_pg_statistic_stavalues1 - 1] = false;

            pfree(freq_datums);
        }

        if (mcv_values)
            pfree(mcv_values);
        if (mcv_freqs)
            pfree(mcv_freqs);
    }

    // Open pg_statistic for update
    Relation sd = table_open(StatisticRelationId, RowExclusiveLock);

    // Check if there's an existing row
    HeapTuple oldtup = SearchSysCache3(STATRELATTINH,
                                        ObjectIdGetDatum(relid),
                                        Int16GetDatum(attnum),
                                        BoolGetDatum(false));

    CatalogIndexState indstate = CatalogOpenIndexes(sd);
    HeapTuple stup;

    if (HeapTupleIsValid(oldtup)) {
        // Update existing row
        stup = heap_modify_tuple(oldtup, RelationGetDescr(sd), values, nulls, replaces);
        ReleaseSysCache(oldtup);
        CatalogTupleUpdateWithInfo(sd, &stup->t_self, stup, indstate);
    } else {
        // Insert new row
        stup = heap_form_tuple(RelationGetDescr(sd), values, nulls);
        CatalogTupleInsertWithInfo(sd, stup, indstate);
    }

    heap_freetuple(stup);
    CatalogCloseIndexes(indstate);
    table_close(sd, RowExclusiveLock);

    elog(DEBUG1, "Injected DeepLake statistics for column %s (null_frac=%.4f, avg_width=%d, n_distinct=%.2f)",
         NameStr(attr->attname), dl_stats.null_frac, dl_stats.avg_width, dl_stats.n_distinct);

    return true;
}

bool inject_deeplake_statistics(Relation rel)
{
    const Oid relid = RelationGetRelid(rel);

    // Check if this is a DeepLake table
    if (!table_storage::instance().table_exists(relid)) {
        return false;
    }

    TupleDesc tupdesc = RelationGetDescr(rel);
    bool any_injected = false;

    for (int16_t attnum = 1; attnum <= tupdesc->natts; ++attnum) {
        Form_pg_attribute attr = TupleDescAttr(tupdesc, attnum - 1);
        if (attr->attisdropped) {
            continue;
        }

        if (inject_column_statistics(rel, attnum)) {
            any_injected = true;
        }
    }

    if (any_injected) {
        elog(DEBUG1, "Injected DeepLake statistics for relation %s", RelationGetRelationName(rel));
    }

    return any_injected;
}

} // namespace pg
