#include "duckdb_deeplake_scan.hpp"

#include <duckdb.hpp>
#include <duckdb/catalog/catalog.hpp>
#include <duckdb/common/types/uuid.hpp>
#include <duckdb/parser/parsed_data/create_table_function_info.hpp>
#include <duckdb/planner/expression/bound_conjunction_expression.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>
#include <duckdb/planner/filter/in_filter.hpp>

#include "duckdb_deeplake_convert.hpp"
#include "pg_deeplake.hpp"
#include "table_data.hpp"
#include "table_storage.hpp"
#include "utils.hpp"

#include <base/function.hpp>
#include <heimdall_common/filtered_dataset.hpp>
#include <query_core/index_holder.hpp>

#include <source_location>

namespace {

struct deeplake_scan_bind_data final : public duckdb::TableFunctionData
{
    pg::table_data& table_data;
    duckdb::vector<duckdb::LogicalType> bind_types;

    deeplake_scan_bind_data(pg::table_data& td_ref, duckdb::vector<duckdb::LogicalType> types)
        : table_data(td_ref)
        , bind_types(std::move(types))
    {
    }

    ~deeplake_scan_bind_data() override
    {
    }
};

struct deeplake_scan_global_state final : public duckdb::GlobalTableFunctionState
{
    duckdb::vector<duckdb::column_t> column_ids;
    std::vector<base::function<async::promise<std::vector<icm::roaring>>()>> index_searchers;
    duckdb::unique_ptr<duckdb::Expression> filter_expr;
    std::mutex index_search_mutex;
    heimdall::dataset_view_ptr index_search_result;
    std::atomic<int64_t> current_row = 0;

    idx_t MaxThreads() const override
    {
        return std::min(base::system_report::cpu_cores(), pg::max_num_threads_for_global_state);
    }
};

struct deeplake_scan_local_state final : public duckdb::LocalTableFunctionState
{
    duckdb::unique_ptr<duckdb::ExpressionExecutor> filter_executor;
};

// Map PostgreSQL type OID to DuckDB LogicalType
duckdb::LogicalType pg_type_to_duckdb(Oid typid, int32_t typmod, int32_t ndims = 1)
{
    using namespace duckdb;
    // Resolve domain types to their base type
    switch (typid) {
    case BOOLOID:
        return LogicalType::BOOLEAN;
    case INT2OID:
        return LogicalType::SMALLINT;
    case INT4OID:
        return LogicalType::INTEGER;
    case INT8OID:
        return LogicalType::BIGINT;
    case FLOAT4OID:
        return LogicalType::FLOAT;
    case FLOAT8OID:
    case NUMERICOID:
        return LogicalType::DOUBLE;
    case DATEOID:
        return LogicalType::DATE;
    case TIMEOID:
        return LogicalType::TIME;
    case TIMESTAMPOID:
        return LogicalType::TIMESTAMP;
    case TIMESTAMPTZOID:
        return LogicalType::TIMESTAMP_TZ;
    case CHAROID:
    case BPCHAROID:
    case VARCHAROID:
    case TEXTOID:
        return LogicalType::VARCHAR;
    case JSONOID:
    case JSONBOID: {
        LogicalType res(LogicalType::VARCHAR);
        res.SetAlias(LogicalType::JSON_TYPE_NAME);
        return res;
    }
    case UUIDOID:
        return LogicalType::UUID;
    case BYTEAOID:
        return LogicalType::BLOB;
    // Array types - map to DuckDB LIST, with proper nesting for multi-dimensional arrays
    case BOOLARRAYOID: {
        LogicalType elem_type = LogicalType::BOOLEAN;
        for (int32_t i = 1; i < ndims; ++i)
            elem_type = LogicalType::LIST(elem_type);
        return LogicalType::LIST(elem_type);
    }
    case INT2ARRAYOID: {
        LogicalType elem_type = LogicalType::SMALLINT;
        for (int32_t i = 1; i < ndims; ++i)
            elem_type = LogicalType::LIST(elem_type);
        return LogicalType::LIST(elem_type);
    }
    case INT4ARRAYOID: {
        LogicalType elem_type = LogicalType::INTEGER;
        for (int32_t i = 1; i < ndims; ++i)
            elem_type = LogicalType::LIST(elem_type);
        return LogicalType::LIST(elem_type);
    }
    case INT8ARRAYOID: {
        LogicalType elem_type = LogicalType::BIGINT;
        for (int32_t i = 1; i < ndims; ++i)
            elem_type = LogicalType::LIST(elem_type);
        return LogicalType::LIST(elem_type);
    }
    case FLOAT4ARRAYOID: {
        LogicalType elem_type = LogicalType::FLOAT;
        for (int32_t i = 1; i < ndims; ++i)
            elem_type = LogicalType::LIST(elem_type);
        return LogicalType::LIST(elem_type);
    }
    case FLOAT8ARRAYOID: {
        LogicalType elem_type = LogicalType::DOUBLE;
        for (int32_t i = 1; i < ndims; ++i)
            elem_type = LogicalType::LIST(elem_type);
        return LogicalType::LIST(elem_type);
    }
    case TEXTARRAYOID:
    case VARCHARARRAYOID: {
        LogicalType elem_type = LogicalType::VARCHAR;
        for (int32_t i = 1; i < ndims; ++i)
            elem_type = LogicalType::LIST(elem_type);
        return LogicalType::LIST(elem_type);
    }
    case DATEARRAYOID: {
        LogicalType elem_type = LogicalType::DATE;
        for (int32_t i = 1; i < ndims; ++i)
            elem_type = LogicalType::LIST(elem_type);
        return LogicalType::LIST(elem_type);
    }
    case TIMESTAMPARRAYOID: {
        LogicalType elem_type = LogicalType::TIMESTAMP;
        for (int32_t i = 1; i < ndims; ++i)
            elem_type = LogicalType::LIST(elem_type);
        return LogicalType::LIST(elem_type);
    }
    case TIMESTAMPTZARRAYOID: {
        LogicalType elem_type = LogicalType::TIMESTAMP_TZ;
        for (int32_t i = 1; i < ndims; ++i)
            elem_type = LogicalType::LIST(elem_type);
        return LogicalType::LIST(elem_type);
    }
    case BYTEAARRAYOID: {
        LogicalType elem_type = LogicalType::BLOB;
        for (int32_t i = 1; i < ndims; ++i)
            elem_type = LogicalType::LIST(elem_type);
        return LogicalType::LIST(elem_type);
    }
    default:
        throw duckdb::NotImplementedException("Unsupported PostgreSQL type OID: " + std::to_string(typid) + " (" +
                                              format_type_with_typemod(typid, typmod) + ")");
    }
}

// Bind function: Define schema from table_data
duckdb::unique_ptr<duckdb::FunctionData> deeplake_scan_bind(duckdb::ClientContext& context,
                                                            duckdb::TableFunctionBindInput& input,
                                                            duckdb::vector<duckdb::LogicalType>& return_types,
                                                            duckdb::vector<duckdb::string>& names)
{
    ASSERT(input.inputs.size() == 1);

    // Extract table_data pointer passed as uint
    const auto table_id = input.inputs[0].GetValue<Oid>();
    ASSERT(table_id != InvalidOid);

    auto& td = pg::table_storage::instance().get_table_data(table_id);

    // Build schema from table_data's tuple descriptor
    for (int32_t i = 0; i < td.num_columns(); ++i) {
        names.emplace_back(td.get_atttypename(i));
        int32_t ndims = td.get_attndims(i);
        // For array types, use attndims; for non-arrays, ndims is 0
        if (ndims == 0 && type_is_array(td.get_base_atttypid(i))) {
            ndims = 1; // Default to 1D if attndims not set but type is array
        }
        return_types.push_back(pg_type_to_duckdb(td.get_base_atttypid(i), td.get_atttypmod(i), ndims));
    }

    return duckdb::make_uniq<deeplake_scan_bind_data>(td, return_types);
}

base::function<async::promise<std::vector<icm::roaring>>()>
try_get_index_searcher(heimdall::column_view_ptr column_view, const duckdb::ConstantFilter& filter)
{
    base::function<async::promise<std::vector<icm::roaring>>()> result;
    auto index_holder = column_view->index_holder();
    ASSERT(index_holder != nullptr);
    auto constant = pg::to_deeplake_value(filter.constant);
    if (nd::dtype_is_numeric(constant.dtype())) {
        query_core::inverted_index_search_info info;
        switch (filter.comparison_type) {
        case duckdb::ExpressionType::COMPARE_EQUAL:
            info.op = query_core::relational_operator::equals;
            break;
        case duckdb::ExpressionType::COMPARE_NOTEQUAL:
            break;
        case duckdb::ExpressionType::COMPARE_LESSTHAN:
            info.op = query_core::relational_operator::less;
            break;
        case duckdb::ExpressionType::COMPARE_GREATERTHAN:
            info.op = query_core::relational_operator::greater;
            break;
        case duckdb::ExpressionType::COMPARE_LESSTHANOREQUALTO:
            info.op = query_core::relational_operator::less_eq;
            break;
        case duckdb::ExpressionType::COMPARE_GREATERTHANOREQUALTO:
            info.op = query_core::relational_operator::greater_eq;
            break;
        default:
            break;
        }
        if (info.op != query_core::relational_operator::invalid) {
            info.column_name = column_view->name();
            info.search_values.push_back(constant);
            if (index_holder->can_run_query(info)) {
                result = [index_holder, si = std::move(info)]() {
                    return index_holder->run_query(si);
                };
            }
        }
    } else if (constant.dtype() == nd::dtype::string) {
        // handle duckdb::ExpressionType::COMPARE_NOTEQUAL ?
        if (filter.comparison_type == duckdb::ExpressionType::COMPARE_EQUAL) {
            query_core::text_search_info info;
            info.column_name = column_view->name();
            info.type = query_core::text_search_info::search_type::equals;
            info.search_values.push_back(std::vector<std::string>{filter.constant.ToString()});
            if (index_holder->can_run_query(info)) {
                result = [index_holder, si = std::move(info)]() {
                    return index_holder->run_query(si);
                };
            }
        }
    }
    return result;
}

base::function<async::promise<std::vector<icm::roaring>>()>
try_get_index_searcher(heimdall::column_view_ptr column_view, const duckdb::InFilter& filter)
{
    query_core::inverted_index_search_info info;
    info.column_name = column_view->name();
    info.op = query_core::relational_operator::equals;
    info.search_values.reserve(filter.values.size());
    for (const duckdb::Value& v : filter.values) {
        info.search_values.push_back(pg::to_deeplake_value(v));
    }
    ASSERT(column_view->index_holder() != nullptr);
    return [h = column_view->index_holder(), si = std::move(info)]() {
        return h->run_query(si);
    };
}

base::function<async::promise<std::vector<icm::roaring>>()>
try_get_index_searcher(heimdall::column_view_ptr column_view, const duckdb::TableFilter& filter)
{
    base::function<async::promise<std::vector<icm::roaring>>()> result;
    ASSERT(column_view != nullptr);
    if (column_view->index_holder() == nullptr) {
        return result;
    }
    switch (filter.filter_type) {
    case duckdb::TableFilterType::CONSTANT_COMPARISON: {
        result = try_get_index_searcher(column_view, filter.Cast<const duckdb::ConstantFilter>());
        break;
    }
    case duckdb::TableFilterType::IN_FILTER: {
        result = try_get_index_searcher(column_view, filter.Cast<const duckdb::InFilter>());
        break;
    }
    default:
        break;
    }
    return result;
}

duckdb::unique_ptr<duckdb::GlobalTableFunctionState> deeplake_scan_init_global(duckdb::ClientContext& context,
                                                                               duckdb::TableFunctionInitInput& input)
{
    auto& bind_data = input.bind_data->Cast<deeplake_scan_bind_data>();
    const auto& td = bind_data.table_data;
    auto r = duckdb::make_uniq<deeplake_scan_global_state>();
    r->column_ids = input.column_ids;

    if (input.filters) {
        duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> filter_exprs;
        for (auto& [output_col_idx, filter] : input.filters->filters) {
            ASSERT(output_col_idx >= 0 && output_col_idx < input.column_ids.size());
            const auto col_idx = input.column_ids[output_col_idx];
            auto is = try_get_index_searcher(td.get_column_view(col_idx), *filter);
            if (is) {
                r->index_searchers.push_back(std::move(is));
            } else {
                duckdb::BoundReferenceExpression column_expr_ref(bind_data.bind_types[col_idx], output_col_idx);
                filter_exprs.push_back(filter->ToExpression(column_expr_ref));
            }
        }
        if (filter_exprs.size() == 1) {
            r->filter_expr = std::move(filter_exprs[0]);
        } else if (filter_exprs.size() > 1) {
            // Create a conjunction from the select list.
            auto c = duckdb::make_uniq<duckdb::BoundConjunctionExpression>(duckdb::ExpressionType::CONJUNCTION_AND);
            for (auto& expr : filter_exprs) {
                c->children.push_back(std::move(expr));
            }
            r->filter_expr = std::move(c);
        }
    }
    return r;
}

duckdb::unique_ptr<duckdb::LocalTableFunctionState>
deeplake_scan_init_local(duckdb::ExecutionContext& context,
                         duckdb::TableFunctionInitInput& input,
                         duckdb::GlobalTableFunctionState* global_state)
{
    auto& global = global_state->Cast<deeplake_scan_global_state>();
    auto r = duckdb::make_uniq<deeplake_scan_local_state>();

    // Handle other filter types similarly
    if (global.filter_expr) {
        r->filter_executor = duckdb::make_uniq<duckdb::ExpressionExecutor>(context.client, *global.filter_expr);
    }

    return r;
}

class deeplake_scan_function_helper
{
    const deeplake_scan_bind_data& bind_data_;
    deeplake_scan_global_state& global_state_;
    deeplake_scan_local_state& local_state_;
    duckdb::DataChunk& output_;

public:
    deeplake_scan_function_helper(duckdb::ClientContext& context,
                                  duckdb::TableFunctionInput& data,
                                  duckdb::DataChunk& output)
        : bind_data_(data.bind_data->Cast<deeplake_scan_bind_data>())
        , global_state_(data.global_state->Cast<deeplake_scan_global_state>())
        , local_state_(data.local_state->Cast<deeplake_scan_local_state>())
        , output_(output)
    {
    }

    void scan()
    {
        while (true) {
            if (INTERRUPTS_PENDING_CONDITION()) {
                return;
            }
            do_scan();
            if (output_.size() == 0 || !local_state_.filter_executor) {
                break;
            }
            duckdb::SelectionVector sel(output_.size());
            const idx_t match_count = local_state_.filter_executor->SelectExpression(output_, sel);
            if (match_count != 0) {
                if (match_count != output_.size()) {
                    output_.Slice(sel, match_count);
                }
                break;
            }
        }
    }

private:
    bool has_index_search() const
    {
        return !global_state_.index_searchers.empty();
    }

    bool is_index_search_done() const
    {
        return global_state_.index_search_result != nullptr;
    }

    bool is_uuid_type(duckdb::column_t col_idx) const
    {
        auto att_type = bind_data_.table_data.get_atttypid(col_idx);
        return att_type == UUIDOID;
    }

    bool is_bytea_type(duckdb::column_t col_idx) const
    {
        auto att_type = bind_data_.table_data.get_base_atttypid(col_idx);
        return att_type == BYTEAOID;
    }

    bool is_array_type(duckdb::column_t col_idx) const
    {
        auto att_type = bind_data_.table_data.get_atttypid(col_idx);
        switch (att_type) {
        case BOOLARRAYOID:
        case INT2ARRAYOID:
        case INT4ARRAYOID:
        case INT8ARRAYOID:
        case FLOAT4ARRAYOID:
        case FLOAT8ARRAYOID:
        case TEXTARRAYOID:
        case VARCHARARRAYOID:
        case DATEARRAYOID:
        case TIMESTAMPARRAYOID:
        case TIMESTAMPTZARRAYOID:
        case BYTEAARRAYOID:
            return true;
        default:
            return false;
        }
    }

    static duckdb::string_t
    add_string(duckdb::Vector& vector, const char* data, duckdb::idx_t len, const std::source_location location)
    {
        auto line_number = static_cast<int32_t>(location.line());
        try {
            return duckdb::StringVector::AddString(vector, data, len);
        } catch (const duckdb::Exception& e) {
            elog(ERROR,
                 "DuckDB exception while adding string '%s', line %d: %s",
                 std::string(std::string_view(data, len)).c_str(),
                 line_number,
                 e.what());
        } catch (const std::exception& e) {
            elog(ERROR,
                 "STD exception while adding string '%s', line %d: %s",
                 std::string(std::string_view(data, len)).c_str(),
                 line_number,
                 e.what());
        } catch (...) {
            elog(ERROR,
                 "Unknown exception while adding string '%s', line %d",
                 std::string(std::string_view(data, len)).c_str(),
                 line_number);
        }
    }

    void set_string_column_output(unsigned output_column_id, nd::array&& samples)
    {
        ASSERT(samples.dtype() == nd::dtype::string);
        auto& output_vector = output_.data[output_column_id];
        pg::impl::string_stream_array_holder string_holder(samples);
        for (duckdb::idx_t row_in_batch = 0; row_in_batch < output_.size(); ++row_in_batch) {
            auto* duckdb_data = duckdb::FlatVector::GetData<duckdb::string_t>(output_vector);
            std::string_view value;
            if (string_holder.is_valid()) {
                value = string_holder.data(row_in_batch);
            } else {
                value = base::string_view_cast<const unsigned char>(samples[row_in_batch].data());
            }
            duckdb_data[row_in_batch] =
                add_string(output_vector, value.data(), value.size(), std::source_location::current());
        }
    }

    void set_uuid_column_output(unsigned output_column_id, nd::array&& samples)
    {
        auto& output_vector = output_.data[output_column_id];
        for (duckdb::idx_t row_in_batch = 0; row_in_batch < output_.size(); ++row_in_batch) {
            auto sample = samples[row_in_batch];
            if (sample.is_none()) {
                duckdb::FlatVector::SetNull(output_vector, row_in_batch, true);
                continue;
            }
            // UUID type - convert string to DuckDB UUID (hugeint/INT128)
            auto value = base::string_view_cast<const unsigned char>(sample.data());
            std::string uuid_str(reinterpret_cast<const char*>(value.data()), value.size());

            // Use DuckDB's UUID::FromString to parse UUID string
            try {
                auto uuid_value = duckdb::UUID::FromString(uuid_str);
                auto* duckdb_data = duckdb::FlatVector::GetData<duckdb::hugeint_t>(output_vector);
                duckdb_data[row_in_batch] = uuid_value;
            } catch (...) {
                // If parsing fails, set to NULL
                duckdb::FlatVector::SetNull(output_vector, row_in_batch, true);
            }
        }
    }

    void set_non_array_column_output(unsigned output_column_id, nd::array&& samples)
    {
        const auto col_idx = global_state_.column_ids[output_column_id];
        auto& output_vector = output_.data[output_column_id];
        for (duckdb::idx_t row_in_batch = 0; row_in_batch < output_.size(); ++row_in_batch) {
            auto sample = samples[row_in_batch];
            if (sample.is_none()) {
                duckdb::FlatVector::SetNull(output_vector, row_in_batch, true);
                continue;
            }
            // Non-array type
            nd::switch_dtype(sample.dtype(), [&]<typename T>() {
                if constexpr (std::is_arithmetic_v<T>) {
                    auto att_type = bind_data_.table_data.get_atttypid(col_idx);
                    if (att_type == VARCHAROID || att_type == CHAROID || att_type == BPCHAROID) {
                        auto* duckdb_data = duckdb::FlatVector::GetData<duckdb::string_t>(output_vector);
                        auto value = *reinterpret_cast<const T*>(sample.data().data());
                        duckdb_data[row_in_batch] = add_string(
                            output_vector, reinterpret_cast<const char*>(&value), 1, std::source_location::current());
                        return;
                    }
                    auto* duckdb_data = duckdb::FlatVector::GetData<T>(output_vector);
                    auto value = sample.data().data();
                    duckdb_data[row_in_batch] = *reinterpret_cast<const T*>(value);
                } else if constexpr (std::is_same_v<T, std::span<const uint8_t>>) {
                    auto value = sample.data();
                    if (value.size() == 0) {
                        duckdb::FlatVector::SetNull(output_vector, row_in_batch, true);
                        return;
                    }
                    auto* duckdb_data = duckdb::FlatVector::GetData<duckdb::string_t>(output_vector);
                    duckdb_data[row_in_batch] = duckdb::StringVector::AddStringOrBlob(
                        output_vector, reinterpret_cast<const char*>(value.data()), value.size());
                } else {
                    auto* duckdb_data = duckdb::FlatVector::GetData<duckdb::string_t>(output_vector);
                    if (sample.dtype() == nd::dtype::object) {
                        if (sample.is_none()) {
                            duckdb::FlatVector::SetNull(output_vector, row_in_batch, true);
                        } else {
                            auto json_str = sample.dict_value(0).serialize();
                            duckdb_data[row_in_batch] = add_string(
                                output_vector, json_str.data(), json_str.size(), std::source_location::current());
                        }
                    } else {
                        auto value = base::string_view_cast<const unsigned char>(sample.data());
                        duckdb_data[row_in_batch] =
                            add_string(output_vector, value.data(), value.size(), std::source_location::current());
                    }
                }
            });
        }
    }

    void set_empty_array_output(duckdb::Vector& output_vector, duckdb::idx_t row_in_batch)
    {
        auto list_entry = duckdb::ListVector::GetEntry(output_vector);
        auto offset = duckdb::ListVector::GetListSize(output_vector);
        duckdb::ListVector::SetListSize(output_vector, offset);
        auto& list_data = duckdb::FlatVector::GetData<duckdb::list_entry_t>(output_vector)[row_in_batch];
        list_data.offset = offset;
        list_data.length = 0;
    }

    void set_2d_array_output(duckdb::Vector& output_vector,
                             duckdb::idx_t row_in_batch,
                             const nd::array& sample,
                             int64_t nrows,
                             int64_t ncols)
    {
        elog(LOG, "set_2d_array_output: row_in_batch=%zu, nrows=%ld, ncols=%ld, dtype=%d",
             row_in_batch, nrows, ncols, static_cast<int>(sample.dtype()));

        // Get the child vector (type: LIST(T))
        auto& child_vec = duckdb::ListVector::GetEntry(output_vector);
        auto child_offset = duckdb::ListVector::GetListSize(output_vector);

        elog(LOG, "  output_vector type=%s, child_vec type=%s, child_offset=%zu",
             output_vector.GetType().ToString().c_str(), child_vec.GetType().ToString().c_str(), child_offset);

        // Reserve space in output_vector for nrows list entries
        duckdb::ListVector::Reserve(output_vector, child_offset + nrows);
        duckdb::ListVector::SetListSize(output_vector, child_offset + nrows);

        // Get the grandchild vector (type: T) - the actual data vector
        auto& grandchild_vec = duckdb::ListVector::GetEntry(child_vec);
        auto grandchild_offset = duckdb::ListVector::GetListSize(child_vec);

        elog(LOG, "  grandchild_vec type=%s, grandchild_offset=%zu",
             grandchild_vec.GetType().ToString().c_str(), grandchild_offset);

        // Reserve space in child_vec for nrows * ncols list entries
        duckdb::ListVector::Reserve(child_vec, grandchild_offset + nrows * ncols);
        duckdb::ListVector::SetListSize(child_vec, grandchild_offset + nrows * ncols);

        // Fill the nested structure
        nd::switch_dtype(sample.dtype(), [&]<typename T>() {
            if constexpr (std::is_arithmetic_v<T>) {
                // Copy actual data to grandchild vector
                auto* data_ptr = duckdb::FlatVector::GetData<T>(grandchild_vec);
                const T* array_data = reinterpret_cast<const T*>(sample.data().data());
                std::memcpy(data_ptr + grandchild_offset, array_data, nrows * ncols * sizeof(T));

                // Log first few values being written
                elog(LOG, "  WRITE: copying %ld elements to grandchild at offset %zu",
                     nrows * ncols, grandchild_offset);
                for (int64_t k = 0; k < std::min(nrows * ncols, (int64_t)6); ++k) {
                    if constexpr (std::is_integral_v<T>) {
                        elog(LOG, "    grandchild[%zu] = %ld", grandchild_offset + k, (long)array_data[k]);
                    } else {
                        elog(LOG, "    grandchild[%zu] = %f", grandchild_offset + k, (double)array_data[k]);
                    }
                }

                // Set up child_vec list entries (one per row, pointing to ranges in grandchild_vec)
                auto* child_entries = duckdb::FlatVector::GetData<duckdb::list_entry_t>(child_vec);
                for (int64_t i = 0; i < nrows; ++i) {
                    child_entries[child_offset + i].offset = grandchild_offset + i * ncols;
                    child_entries[child_offset + i].length = ncols;
                    elog(LOG, "  child_entries[%zu]: offset=%zu, length=%zu",
                         child_offset + i, child_entries[child_offset + i].offset, child_entries[child_offset + i].length);
                }

                // Set up output_vector list entry (pointing to range in child_vec)
                auto* output_entries = duckdb::FlatVector::GetData<duckdb::list_entry_t>(output_vector);
                output_entries[row_in_batch].offset = child_offset;
                output_entries[row_in_batch].length = nrows;
                elog(LOG, "  output_entries[%zu]: offset=%zu, length=%zu",
                     row_in_batch, output_entries[row_in_batch].offset, output_entries[row_in_batch].length);
            } else {
                // String or bytea arrays with 2D structure
                auto* child_entries = duckdb::FlatVector::GetData<duckdb::list_entry_t>(child_vec);
                auto* output_entries = duckdb::FlatVector::GetData<duckdb::list_entry_t>(output_vector);

                for (int64_t i = 0; i < nrows; ++i) {
                    child_entries[child_offset + i].offset = grandchild_offset + i * ncols;
                    child_entries[child_offset + i].length = ncols;

                    for (int64_t j = 0; j < ncols; ++j) {
                        int64_t flat_idx = i * ncols + j;
                        auto elem = sample[flat_idx];

                        if constexpr (std::is_same_v<T, std::span<const uint8_t>>) {
                            auto value = elem.data();
                            auto* data_ptr = duckdb::FlatVector::GetData<duckdb::string_t>(grandchild_vec);
                            data_ptr[grandchild_offset + flat_idx] = duckdb::StringVector::AddStringOrBlob(
                                grandchild_vec, reinterpret_cast<const char*>(value.data()), value.size());
                        } else {
                            auto elem_view = base::string_view_cast<const unsigned char>(elem.data());
                            auto* data_ptr = duckdb::FlatVector::GetData<duckdb::string_t>(grandchild_vec);
                            data_ptr[grandchild_offset + flat_idx] = add_string(
                                grandchild_vec, elem_view.data(), elem_view.size(), std::source_location::current());
                        }
                    }
                }

                output_entries[row_in_batch].offset = child_offset;
                output_entries[row_in_batch].length = nrows;
            }
        });
    }

    void set_1d_array_output(duckdb::Vector& output_vector,
                             duckdb::idx_t row_in_batch,
                             const nd::array& sample,
                             int64_t array_len)
    {
        auto& list_entry_vec = duckdb::ListVector::GetEntry(output_vector);
        auto offset = duckdb::ListVector::GetListSize(output_vector);

        // Reserve space for array elements
        duckdb::ListVector::Reserve(output_vector, offset + array_len);
        duckdb::ListVector::SetListSize(output_vector, offset + array_len);

        // Fill array elements
        nd::switch_dtype(sample.dtype(), [&]<typename T>() {
            if constexpr (std::is_arithmetic_v<T>) {
                auto* list_data = duckdb::FlatVector::GetData<T>(list_entry_vec);
                const T* array_data = reinterpret_cast<const T*>(sample.data().data());
                std::memcpy(list_data + offset, array_data, array_len * sizeof(T));
            } else if constexpr (std::is_same_v<T, std::span<const uint8_t>>) {
                auto* list_data = duckdb::FlatVector::GetData<duckdb::string_t>(list_entry_vec);
                for (int64_t i = 0; i < array_len; ++i) {
                    auto value = sample[i].data();
                    list_data[offset + i] = duckdb::StringVector::AddStringOrBlob(
                        list_entry_vec, reinterpret_cast<const char*>(value.data()), value.size());
                }
            } else {
                // String array
                auto* list_data = duckdb::FlatVector::GetData<duckdb::string_t>(list_entry_vec);
                for (int64_t i = 0; i < array_len; ++i) {
                    auto elem = sample[i];
                    auto elem_view = base::string_view_cast<const unsigned char>(elem.data());
                    list_data[offset + i] =
                        add_string(list_entry_vec, elem_view.data(), elem_view.size(), std::source_location::current());
                }
            }
        });

        // Set list entry metadata
        auto& list_data = duckdb::FlatVector::GetData<duckdb::list_entry_t>(output_vector)[row_in_batch];
        list_data.offset = offset;
        list_data.length = array_len;
    }

    // General N-dimensional array output (for 3D+ arrays)
    // Uses a simplified approach: for 3D arrays, just call set_2d_array_output for each "page"
    void set_nd_array_output(duckdb::Vector& output_vector,
                             duckdb::idx_t row_in_batch,
                             const nd::array& sample,
                             const icm::shape& array_shape)
    {
        auto ndim = array_shape.size();

        if (ndim == 3) {
            // 3D array: treat as array of 2D arrays
            auto num_pages = array_shape[0];
            auto nrows = array_shape[1];
            auto ncols = array_shape[2];

            // Get the child vector (type: LIST(LIST(LIST(T))))
            auto& child_vec = duckdb::ListVector::GetEntry(output_vector);
            auto child_offset = duckdb::ListVector::GetListSize(output_vector);

            // Reserve space for num_pages elements in the top-level list
            duckdb::ListVector::Reserve(output_vector, child_offset + num_pages);
            duckdb::ListVector::SetListSize(output_vector, child_offset + num_pages);

            // Process each page (2D slice)
            auto* child_entries = duckdb::FlatVector::GetData<duckdb::list_entry_t>(output_vector);
            for (int64_t page = 0; page < num_pages; ++page) {
                auto page_array = sample[page];
                auto page_offset = duckdb::ListVector::GetListSize(child_vec);

                // Use set_2d_array_output to handle each page
                set_2d_array_output(child_vec, child_offset + page, page_array, nrows, ncols);

                // The list entry for this page is already set by set_2d_array_output at index (child_offset + page)
                // But we need to adjust it since set_2d_array_output uses row_in_batch directly
                // Actually, set_2d_array_output sets entries[row_in_batch], which is entries[child_offset + page]
                // So it should work correctly
            }

            // Set up the top-level list entry
            auto* output_entries = duckdb::FlatVector::GetData<duckdb::list_entry_t>(output_vector);
            output_entries[row_in_batch].offset = child_offset;
            output_entries[row_in_batch].length = num_pages;

        } else {
            // For 4D+ arrays, use recursive decomposition
            // Treat as array of (N-1)D arrays
            auto first_dim_size = array_shape[0];

            auto& child_vec = duckdb::ListVector::GetEntry(output_vector);
            auto child_offset = duckdb::ListVector::GetListSize(output_vector);

            duckdb::ListVector::Reserve(output_vector, child_offset + first_dim_size);
            duckdb::ListVector::SetListSize(output_vector, child_offset + first_dim_size);

            // Build sub-shape (remove first dimension)
            icm::shape sub_shape(array_shape.data() + 1, array_shape.data() + array_shape.size());

            for (int64_t i = 0; i < first_dim_size; ++i) {
                auto sub_array = sample[i];
                set_nd_array_output(child_vec, child_offset + i, sub_array, sub_shape);
            }

            auto* output_entries = duckdb::FlatVector::GetData<duckdb::list_entry_t>(output_vector);
            output_entries[row_in_batch].offset = child_offset;
            output_entries[row_in_batch].length = first_dim_size;
        }
    }

    void set_array_column_output(unsigned output_column_id, nd::array&& samples)
    {
        auto& output_vector = output_.data[output_column_id];
        for (duckdb::idx_t row_in_batch = 0; row_in_batch < output_.size(); ++row_in_batch) {
            auto sample = samples[row_in_batch];
            if (sample.is_none()) {
                duckdb::FlatVector::SetNull(output_vector, row_in_batch, true);
                continue;
            }

            auto array_shape = sample.shape();
            if (array_shape.size() == 0 || array_shape[0] == 0) {
                set_empty_array_output(output_vector, row_in_batch);
            } else if (array_shape.size() == 1) {
                int64_t array_len = array_shape[0];
                set_1d_array_output(output_vector, row_in_batch, sample, array_len);
            } else if (array_shape.size() == 2) {
                int64_t nrows = array_shape[0];
                int64_t ncols = array_shape[1];
                set_2d_array_output(output_vector, row_in_batch, sample, nrows, ncols);
            } else {
                // 3D+ arrays: use the general N-dimensional handler
                set_nd_array_output(output_vector, row_in_batch, sample, array_shape);
            }
        }
    }

    void set_column_output(unsigned output_column_id, nd::array&& samples)
    {
        const auto col_idx = global_state_.column_ids[output_column_id];
        const bool is_array = is_array_type(col_idx);
        const bool is_uuid = is_uuid_type(col_idx);
        if (!is_array && !is_uuid && samples.dtype() == nd::dtype::string) {
            set_string_column_output(output_column_id, std::move(samples));
        } else if (!is_array && is_uuid) {
            set_uuid_column_output(output_column_id, std::move(samples));
        } else if (!is_array) {
            set_non_array_column_output(output_column_id, std::move(samples));
        } else {
            set_array_column_output(output_column_id, std::move(samples));
        }
    }

    void set_streaming_column_output(unsigned output_column_id, int64_t current_row)
    {
        const auto col_idx = global_state_.column_ids[output_column_id];
        if (is_array_type(col_idx)) {
            throw duckdb::InternalException("Array columns should not have streamers");
        }
        const bool is_uuid = is_uuid_type(col_idx);
        const auto batch_size = output_.size();
        auto& td = bind_data_.table_data;
        auto& output_vector = output_.data[output_column_id];

        auto col_view = td.get_column_view(col_idx);
        nd::switch_dtype(col_view->dtype(), [&]<typename T>() {
            if constexpr (std::is_arithmetic_v<T>) {
                auto att_type = td.get_atttypid(col_idx);
                if (att_type == VARCHAROID || att_type == CHAROID || att_type == BPCHAROID) {
                    auto* duckdb_data = duckdb::FlatVector::GetData<duckdb::string_t>(output_vector);
                    for (duckdb::idx_t row_in_batch = 0; row_in_batch < batch_size; ++row_in_batch) {
                        const int64_t row_idx = current_row + row_in_batch;
                        auto value = td.get_streamers().value<T>(col_idx, row_idx);
                        duckdb_data[row_in_batch] = add_string(
                            output_vector, reinterpret_cast<const char*>(&value), 1, std::source_location::current());
                    }
                    return;
                }
                auto* value_ptr = td.get_streamers().value_ptr<T>(col_idx, current_row);
                std::memcpy(duckdb::FlatVector::GetData<T>(output_vector), value_ptr, batch_size * sizeof(T));
            } else if constexpr (std::is_same_v<T, nd::dict>) {
                auto* duckdb_data = duckdb::FlatVector::GetData<duckdb::string_t>(output_vector);
                for (duckdb::idx_t row_in_batch = 0; row_in_batch < batch_size; ++row_in_batch) {
                    const int64_t row_idx = current_row + row_in_batch;
                    auto sample = td.get_streamers().get_sample(col_idx, row_idx);
                    if (sample.is_none()) {
                        duckdb::FlatVector::SetNull(output_vector, row_in_batch, true);
                    } else {
                        auto json_str = sample.dict_value(0).serialize();
                        duckdb_data[row_in_batch] = add_string(
                            output_vector, json_str.data(), json_str.size(), std::source_location::current());
                    }
                }
            } else {
                for (duckdb::idx_t row_in_batch = 0; row_in_batch < batch_size; ++row_in_batch) {
                    const int64_t row_idx = current_row + row_in_batch;
                    auto value = td.get_streamers().value<std::string_view>(col_idx, row_idx);
                    if (is_uuid) {
                        duckdb::hugeint_t uuid_value;
                        if (!duckdb::UUID::FromString(std::string(value), uuid_value)) {
                            std::string tmp(value);
                            elog(ERROR, "Failed to parse UUID string: %s", tmp.c_str());
                        }
                        auto* duckdb_data = duckdb::FlatVector::GetData<duckdb::hugeint_t>(output_vector);
                        duckdb_data[row_in_batch] = uuid_value;
                    } else {
                        auto* duckdb_data = duckdb::FlatVector::GetData<duckdb::string_t>(output_vector);
                        duckdb_data[row_in_batch] =
                            add_string(output_vector, value.data(), value.size(), std::source_location::current());
                    }
                }
            }
        });
    }

    void do_index_search()
    {
        if (!has_index_search()) {
            return;
        }
        std::lock_guard lock(global_state_.index_search_mutex);
        if (is_index_search_done()) {
            return;
        }
        std::vector<async::promise<icm::roaring>> promises;
        for (auto& is : global_state_.index_searchers) {
            promises.push_back(is().then_any([](std::vector<icm::roaring>&& results) {
                ASSERT(results.size() == 1);
                return std::move(results.front());
            }));
        }
        auto combined_promise = async::combine(std::move(promises)).then_any([](std::vector<icm::roaring>&& results) {
            ASSERT(!results.empty());
            icm::roaring& combined = results[0];
            for (size_t i = 1; i < results.size(); ++i) {
                combined &= results[i];
            }
            return std::move(combined);
        });
        auto indices = combined_promise.get_future().get();
        std::vector<int64_t> indices_vec;
        indices_vec.reserve(indices.cardinality());
        for (auto x : indices) {
            indices_vec.push_back(x);
        }
        global_state_.index_search_result = heimdall_common::create_filtered_dataset(
            bind_data_.table_data.get_read_only_dataset(), icm::index_mapping_t<int64_t>::list(std::move(indices_vec)));
    }

    int64_t next_chunk()
    {
        int64_t num_rows = -1;
        if (has_index_search()) {
            ASSERT(is_index_search_done());
            num_rows = global_state_.index_search_result->num_rows();
        } else {
            num_rows = bind_data_.table_data.num_rows();
        }

        // Determine batch size (DuckDB's standard vector size is 2048)
        constexpr duckdb::idx_t DUCKDB_VECTOR_SIZE = 2048;
        auto current_row = global_state_.current_row.fetch_add(DUCKDB_VECTOR_SIZE);
        if (current_row >= num_rows) {
            output_.SetCardinality(0);
            current_row = -1;
        } else {
            const duckdb::idx_t chunk_size = std::min<duckdb::idx_t>(DUCKDB_VECTOR_SIZE, num_rows - current_row);
            output_.SetCardinality(chunk_size);
        }
        return current_row;
    }

    auto request_range_and_set_column_output(heimdall::column_view_ptr cv, unsigned column_id, int64_t start_row)
    {
        const auto end_row = start_row + output_.size();
        return async::run_on_main([cv, start_row, end_row]() {
            return cv->request_range(start_row, end_row, {});
        }).then([this, column_id](nd::array&& samples) {
            set_column_output(column_id, std::move(samples));
        });
    }

    void do_scan()
    {
        do_index_search();
        const auto current_row = next_chunk();
        if (current_row < 0) {
            return;
        }

        ASSERT(output_.ColumnCount() == global_state_.column_ids.size());
        std::vector<async::promise<void>> column_promises;
        // Fill output vectors column by column using table_data streamers
        for (unsigned i = 0; i < global_state_.column_ids.size(); ++i) {
            const auto col_idx = global_state_.column_ids[i];
            ASSERT(col_idx >= 0 && col_idx < bind_data_.table_data.num_columns());
            // Example query, when column is not referenced but requested by duckdb:
            // SELECT COUNT(*) FROM table_name;
            if (!bind_data_.table_data.is_column_requested(col_idx)) {
                continue;
            }
            auto& output_vector = output_.data[i];
            output_vector.SetVectorType(duckdb::VectorType::FLAT_VECTOR);
            auto& mask = duckdb::FlatVector::Validity(output_vector);

            if (has_index_search()) {
                auto cv = ((*global_state_.index_search_result)[col_idx]).shared_from_this();
                column_promises.emplace_back(request_range_and_set_column_output(cv, i, current_row));
            } else if (bind_data_.table_data.column_has_streamer(col_idx)) {
                set_streaming_column_output(i, current_row);
            } else {
                auto cv = bind_data_.table_data.get_column_view(col_idx);
                column_promises.emplace_back(request_range_and_set_column_output(cv, i, current_row));
            }
        }
        async::combine(std::move(column_promises)).get_future().get();
    }
};

void deeplake_scan_function(duckdb::ClientContext& context, duckdb::TableFunctionInput& data, duckdb::DataChunk& output)
{
    deeplake_scan_function_helper helper(context, data, output);
    try {
        helper.scan();
    } catch (const duckdb::Exception& e) {
        elog(ERROR, "DuckDB exception during Deeplake scan: %s", e.what());
    } catch (const std::exception& e) {
        elog(ERROR, "STD exception during Deeplake scan: %s", e.what());
    } catch (...) {
        elog(ERROR, "Unknown exception during Deeplake scan");
    }
}

// Cardinality function: Return exact row count
duckdb::unique_ptr<duckdb::NodeStatistics> deeplake_scan_cardinality(duckdb::ClientContext& context,
                                                                     const duckdb::FunctionData* bind_data_p)
{
    if (!bind_data_p) {
        return nullptr;
    }

    auto& bind_data = bind_data_p->Cast<deeplake_scan_bind_data>();
    auto row_count = bind_data.table_data.num_rows();

    return duckdb::make_uniq<duckdb::NodeStatistics>(row_count, row_count);
}

duckdb::unique_ptr<duckdb::BaseStatistics> deeplake_scan_column_statistics(duckdb::ClientContext& context,
                                                                           const duckdb::FunctionData* bind_data_p,
                                                                           duckdb::column_t column_index)
{
    if (!bind_data_p) {
        return nullptr;
    }

    auto& bind_data = bind_data_p->Cast<deeplake_scan_bind_data>();
    const auto& td = bind_data.table_data;

    try {
        // Get the DuckDB logical type for this column
        auto duckdb_type = bind_data.bind_types[column_index];

        // Create base statistics for this type
        auto stats = duckdb::BaseStatistics::CreateUnknown(duckdb_type);

        // Get Deeplake column view to extract statistics
        auto col_view = td.get_column_view(column_index);
        auto dtype = col_view->dtype();

        if (nd::dtype_is_numeric(dtype) || !td.is_column_nullable(column_index)) {
            stats.Set(duckdb::StatsInfo::CANNOT_HAVE_NULL_VALUES);
        }

        return stats.ToUnique();
    } catch (const std::exception& e) {
        elog(WARNING, "Failed to generate statistics for column %s", e.what());
        return nullptr;
    }
}

} // unnamed namespace

namespace pg {

// Create and register the deeplake_scan table function
void register_deeplake_scan_function(duckdb::Connection& con)
{
    try {
        // Begin a transaction (required for catalog operations)
        con.BeginTransaction();

        duckdb::TableFunction deeplake_scan("deeplake_scan",
                                            {duckdb::LogicalType::UINTEGER}, // table_id
                                            deeplake_scan_function,
                                            deeplake_scan_bind,
                                            deeplake_scan_init_global,
                                            deeplake_scan_init_local);

        deeplake_scan.projection_pushdown = true;
        deeplake_scan.filter_pushdown = pg::is_filter_pushdown_enabled;
        deeplake_scan.cardinality = deeplake_scan_cardinality;
        deeplake_scan.statistics = deeplake_scan_column_statistics;

        // Register function in DuckDB catalog
        auto& catalog = duckdb::Catalog::GetSystemCatalog(*con.context);
        auto info = duckdb::make_uniq<duckdb::CreateTableFunctionInfo>(deeplake_scan);
        catalog.CreateTableFunction(*con.context, info.get());

        // Commit the transaction
        con.Commit();

    } catch (const std::exception& e) {
        elog(ERROR, "Failed to register deeplake_scan function: %s", e.what());
    }
}

} // namespace pg
