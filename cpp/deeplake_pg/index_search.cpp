// Include libintl.h first to avoid conflicts with PostgreSQL's gettext macro
#include <libintl.h>

#ifdef __cplusplus
extern "C" {
#endif

#include <postgres.h>

#include <access/amapi.h>
#include <commands/defrem.h>
#include <commands/event_trigger.h>
#include <commands/vacuum.h>
#include <utils/selfuncs.h>
#include <utils/typcache.h>

#ifdef __cplusplus
} /// extern "C"
#endif

#include "hybrid_query_merge.hpp"
#include "pg_deeplake.hpp"
#include "table_am.hpp"
#include "table_scan.hpp"
#include "table_storage.hpp"

#include <cstdint>
#include <numeric>
#include <vector>

namespace {

/// Helper function to extract column names from PostgreSQL expression tree
std::vector<std::string> extract_column_names_from_expr(Node* expr)
{
    std::vector<std::string> column_names;
    if (expr == nullptr) {
        return column_names;
    }
    switch (nodeTag(expr)) {
        case T_TypeCast: {
            // Handle TypeCast nodes - extract from the underlying expression
            TypeCast* cast = (TypeCast*)expr;
            std::vector<std::string> sub_names = extract_column_names_from_expr(cast->arg);
            column_names.insert(column_names.end(), sub_names.begin(), sub_names.end());
            break;
        }
        case T_RowExpr: {
            // Handle RowExpr nodes - extract from the args list
            RowExpr* row_expr = (RowExpr*)expr;
            ListCell* lc = nullptr;
            foreach (lc, row_expr->args) {
                Node* arg = (Node*)lfirst(lc);
                std::vector<std::string> sub_names = extract_column_names_from_expr(arg);
                column_names.insert(column_names.end(), sub_names.begin(), sub_names.end());
            }
            break;
        }
        case T_ColumnRef: {
            // Handle ColumnRef nodes - extract field names
            ColumnRef* col_ref = (ColumnRef*)expr;
            ListCell* lc = nullptr;
            foreach (lc, col_ref->fields) {
                Node* field = (Node*)lfirst(lc);
                if (IsA(field, String)) {
                    char* field_name = strVal(field);
                    column_names.push_back(std::string(field_name));
                }
            }
            break;
        }
        default:
            // For other node types, we don't extract column names
            break;
    }

    return column_names;
}

/// Parse index options during index creation
pg::index_options parse_options(IndexStmt* stmt)
{
    pg::index_options opts;
    List* options = stmt->options;
    ListCell* lc = nullptr;
    foreach (lc, options) {
        DefElem* def = (DefElem*)lfirst(lc);
        if (def->arg != nullptr) {
            if (std::strcmp(def->defname, pg::index_type_option_name) == 0) {
                const char* idx_type = defGetString(def);
                opts.index_type = idx_type;
            }
        }
    }
    return opts;
}

/// @name Index search runners
/// @description: Helpers to run different types of index searches
/// @{

/// Check if we should keep memory context switcher as a member variable
struct scan_opaque
{
    std::vector<ItemPointerData>::size_type current_ = 0UL;
    std::vector<std::pair<ItemPointerData, float>> opaque_;
};

query_core::query_result run_index_search(nd::array input_array, std::string func_name, const std::string& column_name, pg::index_info& idx_info)
{
    icm::vector<query_core::expr> args;
    args.emplace_back(query_core::expr::make_column_ref(column_name, std::string{}));
    args.emplace_back(query_core::expr::make_literal_array(std::move(input_array)));
    const bool is_cosine_similarity = (func_name == "COSINE_SIMILARITY");
    query_core::top_k_search_info info(
        query_core::expr{},
        query_core::expr::make_function_ref(std::move(func_name), std::move(args)),
        pg::query_info::current().limit(),
        idx_info.order_type());
    if (!is_cosine_similarity && !idx_info.can_run_query(info, column_name)) {
        elog(ERROR, "Cannot run index search with '%s' for column '%s', function: '%s', please review previous errors."
                    "\nResolution: switch to sequential scan or re-create the index",
                    idx_info.index_name().c_str(), column_name.c_str(), info.order_expr.as_function_ref().get_name().c_str());
    }
    return idx_info.run_query(std::move(info), column_name);
}

query_core::query_result run_embedding_search(ArrayType* comparison, const std::string& column_name, pg::index_info& idx_info)
{
    nd::array input_array = pg::utils::pg_to_nd(comparison, false);
    std::string func_name = (input_array.dimensions() == 1 ? "COSINE_SIMILARITY" : "MAXSIM");
    return run_index_search(std::move(input_array), std::move(func_name), column_name, idx_info);
}

query_core::query_result run_text_search(std::string text_value, const std::string& column_name, pg::index_info& idx_info)
{
    nd::array input_array = nd::adapt(text_value);
    std::string func_name = "BM25_SIMILARITY";
    return run_index_search(std::move(input_array), std::move(func_name), column_name, idx_info);
}

icm::roaring run_numeric_search(nd::array comparison, StrategyNumber strategy, pg::index_info& idx_info)
{
    query_core::inverted_index_search_info info;
    switch (strategy) {
    case BTEqualStrategyNumber:
        info.op = query_core::relational_operator::equals;
        break;
    case BTLessStrategyNumber:
        info.op = query_core::relational_operator::less;
        break;
    case BTGreaterStrategyNumber:
        info.op = query_core::relational_operator::greater;
        break;
    case BTLessEqualStrategyNumber:
        info.op = query_core::relational_operator::less_eq;
        break;
    case BTGreaterEqualStrategyNumber:
        info.op = query_core::relational_operator::greater_eq;
        break;
    default:
        break;
    }
    if (info.op == query_core::relational_operator::invalid || !idx_info.can_run_query(info)) {
        return {};
    }
    info.column_name = idx_info.column_name();
    info.search_values.push_back(std::move(comparison));
    return idx_info.run_query(std::move(info));
}

icm::roaring run_exact_text_search(std::string text_value, StrategyNumber strategy, pg::index_info& idx_info)
{
    if (idx_info.index_type() == deeplake_core::deeplake_index_type::type::bm25) {
        elog(ERROR,
             "BM25 index '%s' does not support exact text search (= or @> operators).\n"
             "Hint: Use 'inverted' or 'exact_text' index_type for equality/contains queries, "
             "or use ORDER BY with <#> operator for BM25 similarity search.",
             idx_info.index_name().c_str());
    }
    query_core::text_search_info info;
    switch (strategy) {
    case BTEqualStrategyNumber:  // Strategy 3: =
        info.type = query_core::text_search_info::search_type::equals;
        break;
    case 7:  // Strategy 7: @>  (contains)
        info.type = query_core::text_search_info::search_type::contains;
        break;
    default:
        elog(ERROR, "Unknown strategy for text search: %d", strategy);
        break;
    }
    if (!idx_info.can_run_query(info)) {
        return {};
    }
    info.column_name = idx_info.column_name();
    info.search_values.emplace_back(icm::vector<std::string>{std::move(text_value)});
    return idx_info.run_query(std::move(info));
}

query_core::query_result run_hybrid_search(ScanKey skey, pg::index_info& idx_info)
{
    query_core::query_result result;
    Datum comparison_value = skey->sk_argument;
    if (!DatumGetPointer(comparison_value)) {
        elog(ERROR, "NULL hybrid_record not supported");
    }
    HeapTupleHeader t = DatumGetHeapTupleHeader(comparison_value);
    /// Look up the composite type
    Oid tuptype = HeapTupleHeaderGetTypeId(t);
    int32_t tuptypmod = HeapTupleHeaderGetTypMod(t);
    TupleDesc tupdesc = lookup_rowtype_tupdesc(tuptype, tuptypmod);

    /// Turn into a HeapTuple
    HeapTupleData tmptup;
    tmptup.t_len = HeapTupleHeaderGetDatumLength(t);
    tmptup.t_data = t;

    /// Extract fields
    Datum values[4];
    bool nulls[4];
    heap_deform_tuple(&tmptup, tupdesc, values, nulls);

    ArrayType* embedding      = DatumGetArrayTypeP(values[0]);
    char* text_value          = TextDatumGetCString(values[1]);
    double embedding_weight   = DatumGetFloat8(values[2]);
    double text_weight        = DatumGetFloat8(values[3]);

    auto emb_result = run_embedding_search(embedding, idx_info.column_names()[0], idx_info);
    auto text_result = run_text_search(text_value, idx_info.column_names()[1], idx_info);

    ReleaseTupleDesc(tupdesc);

    /// Merge results using weighted combination with softmax normalization
    return pg::merge_query_results(emb_result, text_result, embedding_weight, text_weight, pg::query_info::current().limit());
}

/// @}

} // namespace

#ifdef __cplusplus
extern "C" {
#endif

PG_FUNCTION_INFO_V1(handle_index_creation);
PG_FUNCTION_INFO_V1(deeplake_index_handler);
PG_FUNCTION_INFO_V1(deeplake_cosine_similarity);
PG_FUNCTION_INFO_V1(deeplake_vector_lt);
PG_FUNCTION_INFO_V1(deeplake_vector_le);
PG_FUNCTION_INFO_V1(deeplake_vector_eq);
PG_FUNCTION_INFO_V1(deeplake_vector_ne);
PG_FUNCTION_INFO_V1(deeplake_vector_ge);
PG_FUNCTION_INFO_V1(deeplake_vector_gt);
PG_FUNCTION_INFO_V1(deeplake_vector_compare);
PG_FUNCTION_INFO_V1(deeplake_maxsim);
PG_FUNCTION_INFO_V1(deeplake_bm25_similarity_text);
PG_FUNCTION_INFO_V1(deeplake_text_contains);
PG_FUNCTION_INFO_V1(deeplake_hybrid_search);
PG_FUNCTION_INFO_V1(deeplake_jsonb_field_eq);

/// @name Callbacks for AM index operations
/// @{

IndexBuildResult* pg_build(Relation heap, Relation index, IndexInfo* indexInfo)
{
    const auto oid = RelationGetRelid(index);
    pg::pg_index::create_index_info(oid);
    pg::save_index_metadata(oid);
    IndexBuildResult* result = (IndexBuildResult*)palloc(sizeof(IndexBuildResult));
    result->heap_tuples = 0.0;
    result->index_tuples = 0.0;

    return result;
}

void pg_buildempty(Relation index)
{
    base::log_debug(base::log_channel::index, "Build empty called");
}

bool pg_insert(Relation index,
               Datum* values,
               bool* isnull,
               ItemPointer heap_tid,
               Relation heap,
               IndexUniqueCheck check_unique,
               bool index_unchanged,
               IndexInfo* index_info)
{
    const bool is_update = (pg::query_info::current().command_type() == pg::command_type::CMD_UPDATE);
    if (index_unchanged && is_update) {
        return false;
    }

    if (!pg::pg_index::has_indexes()) {
        pg::load_index_metadata();
        ASSERT(pg::pg_index::has_index_info(RelationGetRelid(index)));
    }
    return false;
}

IndexBulkDeleteResult* pg_bulkdelete(IndexVacuumInfo* info,
                                     IndexBulkDeleteResult* stats,
                                     IndexBulkDeleteCallback callback,
                                     void* callback_state)
{
    return nullptr;
}

IndexBulkDeleteResult* pg_vacuumcleanup(IndexVacuumInfo* info, IndexBulkDeleteResult* stats)
{
    return stats;
}

void pg_costestimate(PlannerInfo* root,
                     IndexPath* path,
                     double loop_count,
                     Cost* index_startup_cost,
                     Cost* index_total_cost,
                     Selectivity* index_selectivity,
                     double* index_correlation,
                     double* index_pages)
{
    // Check if this is a BM25 index with unsupported operators
    if (path->indexinfo && path->indexclauses != nullptr) {
        Oid index_oid = path->indexinfo->indexoid;
        if (!pg::pg_index::has_index_info(index_oid)) {
            pg::load_index_metadata();
        }
        if (pg::pg_index::has_index_info(index_oid)) {
            auto& idx_info = pg::pg_index::get_index_info(index_oid);
            auto index_type = idx_info.index_type();

            // Check if BM25 index is being used with equality or contains operators
            if (index_type == deeplake_core::deeplake_index_type::type::bm25) {
                ListCell* lc = nullptr;
                foreach (lc, path->indexclauses) {
                    IndexClause* iclause = (IndexClause*)lfirst(lc);
                    RestrictInfo* rinfo = iclause->rinfo;

                    if (rinfo && IsA(rinfo->clause, OpExpr)) {
                        OpExpr* opexpr = (OpExpr*)rinfo->clause;
                        StrategyNumber strategy = get_op_opfamily_strategy(opexpr->opno,
                                                                           path->indexinfo->opfamily[iclause->indexcol]);

                        // BM25 doesn't support equality (strategy 3) or contains (strategy 7)
                        if (strategy == BTEqualStrategyNumber || strategy == 7) {
                            elog(DEBUG1,
                                 "pg_costestimate: disabling BM25 index %s for unsupported operator (strategy %d)",
                                 idx_info.index_name().c_str(), strategy);
                            // Return very high costs to force sequential scan
                            *index_startup_cost = std::numeric_limits<double>::max();
                            *index_total_cost = std::numeric_limits<double>::max();
                            *index_selectivity = 0;
                            *index_correlation = 0;
                            *index_pages = 0;
                            return;
                        }
                    }
                }
            }

            // Check if exact_text index is being used with BM25 search (ORDER BY)
            if (index_type == deeplake_core::deeplake_index_type::type::exact_text && path->indexorderbys != nullptr) {
                elog(DEBUG1,
                     "pg_costestimate: disabling exact_text index %s for BM25 search (ORDER BY)",
                     idx_info.index_name().c_str());
                // Return very high costs to force sequential scan
                *index_startup_cost = std::numeric_limits<double>::max();
                *index_total_cost = std::numeric_limits<double>::max();
                *index_selectivity = 0;
                *index_correlation = 0;
                *index_pages = 0;
                return;
            }
        }
    }

    // Check if this is a JSONB column index
    bool is_jsonb_index = false;
    if (path->indexinfo && path->indexinfo->indexoid != InvalidOid) {
        Relation indexRel = RelationIdGetRelation(path->indexinfo->indexoid);
        if (RelationIsValid(indexRel)) {
            TupleDesc indexTupDesc = RelationGetDescr(indexRel);
            if (indexTupDesc->natts > 0) {
                Form_pg_attribute attr = TupleDescAttr(indexTupDesc, 0);
                Oid coltype = attr->atttypid;
                if (coltype == JSONBOID) {
                    is_jsonb_index = true;
                }
            }
            RelationClose(indexRel);
        }
    }

    // Force index usage for ORDER BY clauses (similarity/ranking searches) or JSONB indexes
    if (path->indexinfo && (path->indexorderbys != nullptr || path->indexclauses != nullptr || is_jsonb_index)) {
        elog(DEBUG1, "pg_costestimate: forcing index scan (jsonb=%d), OID = %u", is_jsonb_index, path->indexinfo->indexoid);
        *index_startup_cost = -1.0;
        *index_total_cost = -1.0;
        *index_selectivity = -1.0;
        *index_correlation = -1.0;
        *index_pages = -1.0;
        return;
    }

    if (path->indexorderbys == nullptr && path->indexclauses == nullptr && path->indexorderbycols == nullptr) {
        *index_startup_cost = std::numeric_limits<double>::max();
        *index_total_cost = std::numeric_limits<double>::max();
        *index_selectivity = 0;
        *index_correlation = 0;
        *index_pages = 0;
    } else {
        GenericCosts costs;
        MemSet(&costs, 0, sizeof(costs));
        genericcostestimate(root, path, loop_count, &costs);
        *index_startup_cost = costs.indexStartupCost;
        *index_total_cost = costs.indexTotalCost;
        *index_selectivity = costs.indexSelectivity;
        *index_correlation = costs.indexCorrelation;
        *index_pages = .0;
    }
}

bytea* pg_options(Datum reloptions, bool validate)
{
    return nullptr;
}

char* pg_buildphasename(int64 phasenum)
{
    return nullptr;
}

bool pg_validate(Oid opclassoid)
{
    return true;
}

IndexScanDesc pg_beginscan(Relation index, int32_t nkeys, int32_t norderbys)
{
    IndexScanDesc scan = RelationGetIndexScan(index, nkeys, norderbys);

    // Set up tuple descriptor for index-only scans (xs_want_itup)
    // This describes the columns available in the index
    scan->xs_itupdesc = RelationGetDescr(index);

    if (!pg::pg_index::has_indexes() && pg::query_info::current().command_type() != pg::command_type::CMD_UNKNOWN) {
        pg::load_index_metadata();
        ASSERT(pg::pg_index::has_index_info(scan->indexRelation->rd_id));
    }
    return scan;
}

/// Function to collect order by and filter data from deeplake index
void collect_index_data(IndexScanDesc scan, ScanKey keys, int32_t nkeys, ScanKey orderbys, int32_t norderbys)
{
    if (!pg::pg_index::has_indexes()) {
        pg::load_index_metadata();
    }

    auto& idx_info = pg::pg_index::get_index_info(scan->indexRelation->rd_id);
    const bool is_hybrid_index = idx_info.is_hybrid_index();
    Oid column_type = InvalidOid;
    int32_t column_mod = 0;
    if (scan->opaque != nullptr) {
        ereport(ERROR, (errcode(ERRCODE_INTERNAL_ERROR), errmsg("Invalid scan state")));
    }
    if (!is_hybrid_index) {
        Relation heap_rel = scan->heapRelation;
        bool need_close = false;
        // If heapRelation is NULL, open it ourselves
        if (heap_rel == nullptr) {
            Oid heap_oid = scan->indexRelation->rd_index->indrelid;
            heap_rel = RelationIdGetRelation(heap_oid);
            need_close = true;
            if (!RelationIsValid(heap_rel)) {
                ereport(ERROR, (errcode(ERRCODE_INTERNAL_ERROR),
                               errmsg("Could not open heap relation %u", heap_oid)));
            }
        }
        TupleDesc tuple_desc = RelationGetDescr(heap_rel);
        /// Get column index (1-based)
        const auto column_index = scan->indexRelation->rd_index->indkey.values[0];
        if (column_index < 1 || column_index > tuple_desc->natts) {
            if (need_close) {
                RelationClose(heap_rel);
            }
            ereport(ERROR, (errcode(ERRCODE_INTERNAL_ERROR), errmsg("Invalid column index")));
        }
        Form_pg_attribute attr = TupleDescAttr(tuple_desc, column_index - 1);
        column_type = pg::utils::get_base_type(attr->atttypid);
        column_mod = attr->atttypmod;
        if (need_close) {
            RelationClose(heap_rel);
        }
        if (column_type == TEXTARRAYOID) {
            ereport(ERROR, (errcode(ERRCODE_INTERNAL_ERROR), errmsg("Text array type not supported")));
        }
    }

    auto state = static_cast<scan_opaque*>(palloc(sizeof(scan_opaque)));
    new (state) scan_opaque();
    scan->opaque = state;

    try {
        for (int32_t i = 0; i < norderbys; ++i) {
            ScanKey skey = &(orderbys[i]);
            Datum comparison_value = skey->sk_argument;
            if (comparison_value == (Datum)0) {
                continue; /// Skip null values
            }
            query_core::query_result result;
            if (is_hybrid_index) {
                result = run_hybrid_search(skey, idx_info);
            } else if (column_type == TEXTOID) {
                if (idx_info.index_type() == deeplake_core::deeplake_index_type::type::exact_text) {
                    elog(ERROR, "Exact text index cannot be used for BM25 search.");
                }
                std::string search_text = TextDatumGetCString(comparison_value);
                result = run_text_search(std::move(search_text), idx_info.column_name(), idx_info);
            } else { /// embedding index
                ArrayType* comparison = DatumGetArrayTypeP(comparison_value);
                result = run_embedding_search(comparison, idx_info.column_name(), idx_info);
            }
            idx_info.collect_result(result, state->opaque_, !is_hybrid_index);
        }
        icm::roaring result;
        for (int32_t i = 0; i < nkeys; ++i) {
            ScanKey skey = &(keys[i]);
            Datum comparison_value = skey->sk_argument;
            if (comparison_value == (Datum)0) {
                continue; /// Skip null values
            }
            auto arr = pg::utils::datum_to_nd(comparison_value, column_type, column_mod);
            if (nd::dtype_is_numeric(arr.dtype())) {
                if (i == 0) {
                    result = run_numeric_search(std::move(arr), skey->sk_strategy, idx_info);
                } else {
                    result &= run_numeric_search(std::move(arr), skey->sk_strategy, idx_info);
                }
            } else if (column_type == TEXTOID) {
                std::string search_text = TextDatumGetCString(comparison_value);
                if (i == 0) {
                    result = run_exact_text_search(std::move(search_text), skey->sk_strategy, idx_info);
                } else {
                    result &= run_exact_text_search(std::move(search_text), skey->sk_strategy, idx_info);
                }
            } else if (column_type == JSONBOID) {
                // Handle JSONB @> containment queries (transformed from ->>)
                // Extract field value from the JSONB constant like {"kind": "commit"}
                Jsonb* search_jsonb = DatumGetJsonbP(comparison_value);

                // Convert JSONB to JSON string and parse it
                Datum json_text = DirectFunctionCall1(jsonb_out, JsonbPGetDatum(search_jsonb));
                char* json_str = DatumGetCString(json_text);
                elog(INFO, "Still unprocessed JSONB containment search value: %s", json_str);
            } else {
                ereport(ERROR,
                        (errcode(ERRCODE_INTERNAL_ERROR),
                         errmsg("Unsupported column type for index search: '%s'", format_type_be(column_type))));
            }
        }
        if (nkeys > 0) {
            icm::vector<int64_t> row_numbers;
            row_numbers.reserve(result.cardinality());
            std::transform(result.begin(), result.end(), std::back_inserter(row_numbers), [](auto v) {
                return static_cast<int64_t>(v);
            });
            idx_info.collect_result(query_core::query_result{icm::index_mapping_t<int64_t>::list(std::move(row_numbers))}, state->opaque_, !is_hybrid_index);
        }
    } catch (const std::exception& e) {
        ereport(ERROR, (errcode(ERRCODE_INTERNAL_ERROR), errmsg("%s", e.what())));
    }
}

void pg_rescan(IndexScanDesc scan, ScanKey keys, int32_t nkeys, ScanKey orderbys, int32_t norderbys)
{
    collect_index_data(scan, keys, nkeys, orderbys, norderbys);
}

void pg_endscan(IndexScanDesc scan)
{
    if (scan->opaque != nullptr) {
        auto state = static_cast<scan_opaque*>(scan->opaque);
        pfree(state);
        scan->opaque = nullptr;
    }
}

bool pg_canreturn(Relation index_relation, int32_t attno)
{
    return true;
}

/// Bitmap scan: collect all matching TIDs into a bitmap
/// Returns the number of TIDs added to the bitmap
int64 pg_getbitmap(IndexScanDesc scan, TIDBitmap* tbm)
{
    if (scan == nullptr || tbm == nullptr || scan->opaque == nullptr) {
        base::log_warning(base::log_channel::index, "pg_getbitmap called with uninitialized scan state");
        return 0;
    }
    auto state = static_cast<scan_opaque*>(scan->opaque);
    for (size_t i = 0; i < state->opaque_.size(); i++) {
        ItemPointerData tid = state->opaque_[i].first;
        tbm_add_tuples(tbm, &tid, 1, false);
    }
    return state->opaque_.size();
}

bool pg_gettuple(IndexScanDesc scan, ScanDirection dir)
{
    if (scan == nullptr || scan->opaque == nullptr) {
        return false;
    }
    auto state = static_cast<scan_opaque*>(scan->opaque);
    if (state->current_ >= state->opaque_.size()) {
        return false;
    }
    pg::query_info::current().set_current_score(state->opaque_[state->current_].second);
    const auto block_id = ItemPointerGetBlockNumber(&state->opaque_[state->current_].first);
    const auto offset = ItemPointerGetOffsetNumber(&state->opaque_[state->current_].first);
    if (scan->xs_want_itup) { /// Index only scan
        // Fetch the actual column data from the heap table
        auto table_id = RelationGetRelid(scan->heapRelation);
        pg::table_scan table_scan(table_id, false, false);
        const auto row_number = pg::utils::tid_to_row_number(&state->opaque_[state->current_].first);
        table_scan.set_current_position(row_number);

        // Build an IndexTuple with the indexed column values
        // We need to allocate space for IndexTupleData header + column data
        std::vector<Datum> values(scan->xs_itupdesc->natts);
        std::vector<int8_t> nulls(scan->xs_itupdesc->natts);

        // Fetch values for each indexed column
        for (int32_t i = 0; i < scan->xs_itupdesc->natts; ++i) {
            // Get the heap table column number for this index column (1-based)
            // rd_index->indkey.values[i] contains the heap column number for index column i
            AttrNumber heap_attno = scan->indexRelation->rd_index->indkey.values[i];

            if (heap_attno <= 0) {
                // Expression index column - not supported yet
                elog(ERROR, "Expression index columns not supported in index-only scan");
            }

            // Convert to 0-based for get_datum
            int32_t heap_col_idx = heap_attno - 1;
            auto [datum, is_null] = table_scan.get_datum(heap_col_idx, row_number);
            values[i] = datum;
            nulls[i] = static_cast<int8_t>(is_null);
        }

        // Form the index tuple
        scan->xs_itup = index_form_tuple(scan->xs_itupdesc, values.data(), reinterpret_cast<bool*>(nulls.data()));
        ItemPointerSet(&scan->xs_itup->t_tid, block_id, offset);

        ++state->current_;
        scan->xs_recheck = false;
        scan->xs_recheckorderby = false;
        return true;
    }
    ++state->current_;
    ItemPointerSet(&scan->xs_heaptid, block_id, offset);
    scan->xs_recheck = false;
    scan->xs_recheckorderby = false;
    return true;
}

/// @}

/// @name: Exported functions
/// @description: These functions are exported to PostgreSQL
/// @{

/// Function to be called when CREATE INDEX is triggered
PGDLLEXPORT Datum handle_index_creation(PG_FUNCTION_ARGS)
{
    if (!CALLED_AS_EVENT_TRIGGER(fcinfo)) {
        ereport(ERROR, (errcode(ERRCODE_INTERNAL_ERROR), errmsg("Not fired by event trigger manager")));
    }
    EventTriggerData* trigdata = (EventTriggerData*)fcinfo->context;
    if (nodeTag(trigdata->parsetree) != T_IndexStmt || std::strcmp(GetCommandTagName(trigdata->tag), "CREATE INDEX") != 0) {
        PG_RETURN_VOID();
    }
    IndexStmt* stmt = (IndexStmt*)trigdata->parsetree;
    if (stmt->accessMethod != nullptr && std::strcmp(stmt->accessMethod, "deeplake_index") != 0) {
        PG_RETURN_VOID();
    }
    pg::init_deeplake();
    pg::index_info::current().reset();
    std::string full_name = (stmt->relation->schemaname != nullptr) ? stmt->relation->schemaname : "public";
    std::string table_name = stmt->relation->relname;
    std::string index_name = stmt->idxname;
    full_name += ("." + table_name);
    if (!pg::table_storage::instance().table_exists(full_name)) {
        elog(ERROR, "Table '%s' does not have DeepLake access method, can't create index.", full_name.c_str());
    }

    auto opts = ::parse_options(stmt);
    pg::index_info::current().set_table_name(full_name);
    pg::index_info::current().set_index_name(index_name);
    if (!opts.index_type.empty()) {
        pg::index_info::current().set_index_type(opts.index_type);
    }

    ListCell* lc = nullptr;
    foreach (lc, stmt->indexParams) {
        IndexElem* index_elem = (IndexElem*)lfirst(lc);
        if (index_elem->ordering == SORTBY_DESC) {
            pg::index_info::current().set_order_type(query_core::order_type::descending);
        } else if (index_elem->ordering == SORTBY_ASC) {
            pg::index_info::current().set_order_type(query_core::order_type::ascending);
        }

        const char* column_name = index_elem->name;
        if (column_name != nullptr) {
            pg::index_info::current().set_column_name(column_name);
        } else if (index_elem->expr != nullptr) {
            // Extract column names from the expression tree
            std::vector<std::string> column_names = extract_column_names_from_expr(index_elem->expr);
            if (!column_names.empty()) {
                pg::index_info::current().set_column_names(std::move(column_names));
            } else {
                elog(ERROR, "Failed to extract column names from expression");
            }
        } else {
            elog(ERROR, "Index Column name and expression are both null");
        }
    }

    PG_RETURN_VOID();
}

PGDLLEXPORT Datum deeplake_cosine_similarity(PG_FUNCTION_ARGS)
{
    if (pg::query_info::current().has_current_score()) {
        auto score = pg::query_info::current().current_score();
        pg::query_info::current().reset_current_score();
        PG_RETURN_FLOAT4(score);
    }
    return pg::utils::pg_try([&] {
        auto v1 = pg::utils::pg_to_nd(PG_GETARG_ARRAYTYPE_P(0), false);
        auto v2 = pg::utils::pg_to_nd(PG_GETARG_ARRAYTYPE_P(1), false);
        float result = 0.0f;
        if (v1.dimensions() == 2) {
            result = pg::pg_index::instance().maxsim(v1, v2);
        } else {
            result = pg::pg_index::instance().deeplake_cosine_similarity(v1, v2);
        }
        PG_RETURN_FLOAT4(result);
    });
}

PGDLLEXPORT Datum deeplake_vector_lt(PG_FUNCTION_ARGS)
{
    return pg::utils::pg_try([&] {
        auto v1 = pg::utils::pg_to_nd(PG_GETARG_ARRAYTYPE_P(0), false);
        auto v2 = pg::utils::pg_to_nd(PG_GETARG_ARRAYTYPE_P(1), false);
        bool result = pg::pg_index::instance().vector_lt(v1, v2);
        PG_RETURN_BOOL(result);
    });
}

PGDLLEXPORT Datum deeplake_vector_le(PG_FUNCTION_ARGS)
{
    return pg::utils::pg_try([&] {
        auto v1 = pg::utils::pg_to_nd(PG_GETARG_ARRAYTYPE_P(0), false);
        auto v2 = pg::utils::pg_to_nd(PG_GETARG_ARRAYTYPE_P(1), false);
        bool result = pg::pg_index::instance().vector_le(v1, v2);
        PG_RETURN_BOOL(result);
    });
}

PGDLLEXPORT Datum deeplake_vector_eq(PG_FUNCTION_ARGS)
{
    return pg::utils::pg_try([&] {
        auto v1 = pg::utils::pg_to_nd(PG_GETARG_ARRAYTYPE_P(0), false);
        auto v2 = pg::utils::pg_to_nd(PG_GETARG_ARRAYTYPE_P(1), false);
        const bool result = pg::pg_index::instance().vector_eq(v1, v2);
        PG_RETURN_BOOL(result);
    });
}

PGDLLEXPORT Datum deeplake_vector_ne(PG_FUNCTION_ARGS)
{
    return pg::utils::pg_try([&] {
        auto v1 = pg::utils::pg_to_nd(PG_GETARG_ARRAYTYPE_P(0), false);
        auto v2 = pg::utils::pg_to_nd(PG_GETARG_ARRAYTYPE_P(1), false);
        bool result = pg::pg_index::instance().vector_ne(v1, v2);
        PG_RETURN_BOOL(result);
    });
}

PGDLLEXPORT Datum deeplake_vector_ge(PG_FUNCTION_ARGS)
{
    return pg::utils::pg_try([&] {
        auto v1 = pg::utils::pg_to_nd(PG_GETARG_ARRAYTYPE_P(0), false);
        auto v2 = pg::utils::pg_to_nd(PG_GETARG_ARRAYTYPE_P(1), false);
        bool result = pg::pg_index::instance().vector_ge(v1, v2);
        PG_RETURN_BOOL(result);
    });
}

PGDLLEXPORT Datum deeplake_vector_gt(PG_FUNCTION_ARGS)
{
    return pg::utils::pg_try([&] {
        auto v1 = pg::utils::pg_to_nd(PG_GETARG_ARRAYTYPE_P(0), false);
        auto v2 = pg::utils::pg_to_nd(PG_GETARG_ARRAYTYPE_P(1), false);
        bool result = pg::pg_index::instance().vector_gt(v1, v2);
        PG_RETURN_BOOL(result);
    });
}

PGDLLEXPORT Datum deeplake_vector_compare(PG_FUNCTION_ARGS)
{
    return pg::utils::pg_try([&] {
        auto v1 = pg::utils::pg_to_nd(PG_GETARG_ARRAYTYPE_P(0), false);
        auto v2 = pg::utils::pg_to_nd(PG_GETARG_ARRAYTYPE_P(1), false);
        int64_t result = pg::pg_index::instance().vector_compare(v1, v2);
        PG_RETURN_INT32(result);
    });
}

PGDLLEXPORT Datum deeplake_maxsim(PG_FUNCTION_ARGS)
{
    if (pg::query_info::current().has_current_score()) {
        auto score = pg::query_info::current().current_score();
        pg::query_info::current().reset_current_score();
        PG_RETURN_FLOAT4(score);
    }
    return pg::utils::pg_try([&] {
        auto v1 = pg::utils::pg_to_nd(PG_GETARG_ARRAYTYPE_P(0), false);
        auto v2 = pg::utils::pg_to_nd(PG_GETARG_ARRAYTYPE_P(1), false);
        double result = pg::pg_index::instance().maxsim(v1, v2);
        PG_RETURN_FLOAT4(result);
    });
}

PGDLLEXPORT Datum deeplake_bm25_similarity_text(PG_FUNCTION_ARGS)
{
    if (pg::query_info::current().has_current_score()) {
        auto score = pg::query_info::current().current_score();
        pg::query_info::current().reset_current_score();
        PG_RETURN_FLOAT4(score);
    }
    // Stub: always return 0.0 for now
    PG_RETURN_FLOAT4(0.0);
}

PGDLLEXPORT Datum deeplake_text_contains(PG_FUNCTION_ARGS)
{
#ifdef __USE_GNU
    text* left_text = PG_GETARG_TEXT_PP(0);
    text* right_text = PG_GETARG_TEXT_PP(1);
    const char* left = VARDATA_ANY(left_text);
    const char* right = VARDATA_ANY(right_text);
    const int32 left_len = VARSIZE_ANY_EXHDR(left_text);
    const int32 right_len = VARSIZE_ANY_EXHDR(right_text);
    const bool found = ((left_len >= right_len) && memmem(left, left_len, right, right_len) != nullptr);
    PG_RETURN_BOOL(found);
#else
    const char* left = text_to_cstring(PG_GETARG_TEXT_P(0));
    const char* right = text_to_cstring(PG_GETARG_TEXT_P(1));
    PG_RETURN_BOOL(strstr(left, right) != nullptr);
#endif
}

PGDLLEXPORT Datum deeplake_hybrid_search(PG_FUNCTION_ARGS)
{
    if (pg::query_info::current().has_current_score()) {
        auto score = pg::query_info::current().current_score();
        pg::query_info::current().reset_current_score();
        PG_RETURN_FLOAT4(score);
    }
    PG_RETURN_FLOAT4(0.0);
}

PGDLLEXPORT Datum deeplake_jsonb_field_eq(PG_FUNCTION_ARGS)
{
    Jsonb* jb = PG_GETARG_JSONB_P(0);
    text* field_text = PG_GETARG_TEXT_PP(1);
    text* value_text = PG_GETARG_TEXT_PP(2);

    // Extract field name
    char* field_name = text_to_cstring(field_text);

    // Use jsonb_object_field_text to extract the field value
    Datum field_value = DirectFunctionCall2(
        jsonb_object_field_text,
        JsonbPGetDatum(jb),
        PointerGetDatum(field_text)
    );

    if (field_value == (Datum)0) {
        PG_RETURN_BOOL(false);
    }

    // Compare extracted value with the search value
    text* extracted_text = DatumGetTextPP(field_value);
    char* extracted_str = text_to_cstring(extracted_text);
    char* search_str = text_to_cstring(value_text);

    bool result = (strcmp(extracted_str, search_str) == 0);

    pfree(field_name);
    pfree(extracted_str);
    pfree(search_str);

    PG_RETURN_BOOL(result);
}

PGDLLEXPORT Datum deeplake_index_handler(PG_FUNCTION_ARGS)
{
    pg::init_deeplake();
    IndexAmRoutine* amroutine = makeNode(IndexAmRoutine);
    ASSERT(amroutine != nullptr);

    amroutine->amstrategies = 0;
    amroutine->amsupport = 2;
    amroutine->amoptsprocnum = 0;
    amroutine->amcanorder = true;
    amroutine->amcanorderbyop = true;
    amroutine->amcanbackward = true;
    amroutine->amcanunique = false;
    amroutine->amcanmulticol = true;
    amroutine->amoptionalkey = true;
    amroutine->amsearcharray = false;
    amroutine->amsearchnulls = false;
    amroutine->amstorage = false;
    amroutine->amclusterable = false;
    amroutine->ampredlocks = false;
    amroutine->amcanparallel = false;
    amroutine->amusemaintenanceworkmem = false;
    amroutine->amcaninclude = true;
    amroutine->amsummarizing = false;
    amroutine->amkeytype = InvalidOid;
    amroutine->amparallelvacuumoptions = VACUUM_OPTION_PARALLEL_BULKDEL;

    amroutine->ambuild = pg_build;
    amroutine->ambuildempty = pg_buildempty;
    amroutine->aminsert = pg_insert;
    amroutine->ambulkdelete = pg_bulkdelete;
    amroutine->amvacuumcleanup = pg_vacuumcleanup;
    amroutine->amcostestimate = pg_costestimate;
    amroutine->amoptions = pg_options;
    amroutine->ambuildphasename = pg_buildphasename;
    amroutine->amvalidate = pg_validate;
    amroutine->ambeginscan = pg_beginscan;
    amroutine->amrescan = pg_rescan;
    amroutine->amgettuple = pg_gettuple;
    amroutine->amendscan = pg_endscan;

    amroutine->amcanreturn = pg_canreturn;
    amroutine->amproperty = nullptr;
    amroutine->amadjustmembers = nullptr;
    amroutine->amgetbitmap = pg_getbitmap;
    amroutine->ammarkpos = nullptr;
    amroutine->amrestrpos = nullptr;

    // Disable parallel index scan
    amroutine->amestimateparallelscan = nullptr;
    amroutine->aminitparallelscan = nullptr;
    amroutine->amparallelrescan = nullptr;

    PG_RETURN_POINTER(amroutine);
}

/// @}

#ifdef __cplusplus
} /// extern "C"
#endif
