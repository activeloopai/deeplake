#include <duckdb.hpp>

#include "duckdb_deeplake_convert.hpp"
#include "duckdb_pg_convert.hpp"
#include "utils.hpp"

extern "C" {
#include <postgres.h>
#include <catalog/pg_type_d.h>
#include <utils/array.h>
#include <utils/builtins.h>
#include <utils/date.h>
#include <utils/elog.h>
#include <utils/jsonb.h>
#include <utils/lsyscache.h>
#include <utils/timestamp.h>
#include <utils/uuid.h>
}

namespace pg {

Datum duckdb_value_to_pg_datum(
    const duckdb::Vector& vec, size_t row, Oid target_type, int32_t attr_typmod, bool& is_null)
{
    using namespace duckdb;

    // Check for NULL values
    auto& validity = FlatVector::Validity(vec);
    if (!validity.RowIsValid(row)) {
        is_null = true;
        return (Datum)0;
    }

    is_null = false;
    const auto& duckdb_type = vec.GetType();

    try {
        switch (duckdb_type.id()) {
        case LogicalTypeId::BOOLEAN: {
            auto* data = FlatVector::GetData<bool>(vec);
            return BoolGetDatum(data[row]);
        }

        case LogicalTypeId::TINYINT: {
            auto* data = FlatVector::GetData<int8_t>(vec);
            switch (target_type) {
            case INT2OID:
                return Int16GetDatum(static_cast<int16_t>(data[row]));
            case INT4OID:
                return Int32GetDatum(static_cast<int32_t>(data[row]));
            case INT8OID:
                return Int64GetDatum(static_cast<int64_t>(data[row]));
            case FLOAT4OID:
                return Float4GetDatum(static_cast<float>(data[row]));
            case FLOAT8OID:
                return Float8GetDatum(static_cast<double>(data[row]));
            case NUMERICOID:
                return DirectFunctionCall1(int2_numeric, Int16GetDatum(static_cast<int16_t>(data[row])));
            default:
                elog(ERROR, "Unsupported target type %u for TINYINT", target_type);
                return (Datum)0;
            }
        }

        case LogicalTypeId::SMALLINT: {
            auto* data = FlatVector::GetData<int16_t>(vec);
            switch (target_type) {
            case INT2OID:
                return Int16GetDatum(data[row]);
            case INT4OID:
                return Int32GetDatum(static_cast<int32_t>(data[row]));
            case INT8OID:
                return Int64GetDatum(static_cast<int64_t>(data[row]));
            case FLOAT4OID:
                return Float4GetDatum(static_cast<float>(data[row]));
            case FLOAT8OID:
                return Float8GetDatum(static_cast<double>(data[row]));
            case NUMERICOID:
                return DirectFunctionCall1(int2_numeric, Int16GetDatum(data[row]));
            default:
                elog(ERROR, "Unsupported target type %u for SMALLINT", target_type);
                return (Datum)0;
            }
        }

        case LogicalTypeId::INTEGER: {
            auto* data = FlatVector::GetData<int32_t>(vec);
            switch (target_type) {
            case INT2OID:
                return Int16GetDatum(static_cast<int16_t>(data[row]));
            case INT4OID:
                return Int32GetDatum(data[row]);
            case INT8OID:
                return Int64GetDatum(static_cast<int64_t>(data[row]));
            case FLOAT4OID:
                return Float4GetDatum(static_cast<float>(data[row]));
            case FLOAT8OID:
                return Float8GetDatum(static_cast<double>(data[row]));
            case NUMERICOID:
                return DirectFunctionCall1(int4_numeric, Int32GetDatum(data[row]));
            default:
                elog(ERROR, "Unsupported target type %u for INTEGER", target_type);
                return (Datum)0;
            }
        }

        case LogicalTypeId::BIGINT: {
            auto* data = FlatVector::GetData<int64_t>(vec);
            switch (target_type) {
            case INT2OID:
                return Int16GetDatum(static_cast<int16_t>(data[row]));
            case INT4OID:
                return Int32GetDatum(static_cast<int32_t>(data[row]));
            case INT8OID:
                return Int64GetDatum(data[row]);
            case FLOAT4OID:
                return Float4GetDatum(static_cast<float>(data[row]));
            case FLOAT8OID:
                return Float8GetDatum(static_cast<double>(data[row]));
            case NUMERICOID:
                return DirectFunctionCall1(int8_numeric, Int64GetDatum(data[row]));
            default:
                elog(ERROR, "Unsupported target type %u for BIGINT", target_type);
                return (Datum)0;
            }
        }

        case LogicalTypeId::UTINYINT: {
            auto* data = FlatVector::GetData<uint8_t>(vec);
            switch (target_type) {
            case INT2OID:
                return Int16GetDatum(static_cast<int16_t>(data[row]));
            case INT4OID:
                return Int32GetDatum(static_cast<int32_t>(data[row]));
            case INT8OID:
                return Int64GetDatum(static_cast<int64_t>(data[row]));
            case FLOAT4OID:
                return Float4GetDatum(static_cast<float>(data[row]));
            case FLOAT8OID:
                return Float8GetDatum(static_cast<double>(data[row]));
            default:
                elog(ERROR, "Unsupported target type %u for UTINYINT", target_type);
                return (Datum)0;
            }
        }

        case LogicalTypeId::USMALLINT: {
            auto* data = FlatVector::GetData<uint16_t>(vec);
            switch (target_type) {
            case INT2OID:
                return Int16GetDatum(static_cast<int16_t>(data[row]));
            case INT4OID:
                return Int32GetDatum(static_cast<int32_t>(data[row]));
            case INT8OID:
                return Int64GetDatum(static_cast<int64_t>(data[row]));
            case FLOAT4OID:
                return Float4GetDatum(static_cast<float>(data[row]));
            case FLOAT8OID:
                return Float8GetDatum(static_cast<double>(data[row]));
            default:
                elog(ERROR, "Unsupported target type %u for USMALLINT", target_type);
                return (Datum)0;
            }
        }

        case LogicalTypeId::UINTEGER: {
            auto* data = FlatVector::GetData<uint32_t>(vec);
            switch (target_type) {
            case INT2OID:
                return Int16GetDatum(static_cast<int16_t>(data[row]));
            case INT4OID:
                return Int32GetDatum(static_cast<int32_t>(data[row]));
            case INT8OID:
                return Int64GetDatum(static_cast<int64_t>(data[row]));
            case FLOAT4OID:
                return Float4GetDatum(static_cast<float>(data[row]));
            case FLOAT8OID:
                return Float8GetDatum(static_cast<double>(data[row]));
            default:
                elog(ERROR, "Unsupported target type %u for UINTEGER", target_type);
                return (Datum)0;
            }
        }

        case LogicalTypeId::UBIGINT: {
            auto* data = FlatVector::GetData<uint64_t>(vec);
            switch (target_type) {
            case INT2OID:
                return Int16GetDatum(static_cast<int16_t>(data[row]));
            case INT4OID:
                return Int32GetDatum(static_cast<int32_t>(data[row]));
            case INT8OID:
                return Int64GetDatum(static_cast<int64_t>(data[row]));
            case FLOAT4OID:
                return Float4GetDatum(static_cast<float>(data[row]));
            case FLOAT8OID:
                return Float8GetDatum(static_cast<double>(data[row]));
            default:
                elog(ERROR, "Unsupported target type %u for UBIGINT", target_type);
                return (Datum)0;
            }
        }

        case LogicalTypeId::FLOAT: {
            auto* data = FlatVector::GetData<float>(vec);
            switch (target_type) {
            case FLOAT4OID:
                return Float4GetDatum(data[row]);
            case FLOAT8OID:
                return Float8GetDatum(static_cast<double>(data[row]));
            case NUMERICOID:
                return DirectFunctionCall1(float4_numeric, Float4GetDatum(data[row]));
            default:
                elog(ERROR, "Unsupported target type %u for FLOAT", target_type);
                return (Datum)0;
            }
        }

        case LogicalTypeId::DOUBLE: {
            auto* data = FlatVector::GetData<double>(vec);
            switch (target_type) {
            case FLOAT4OID:
                return Float4GetDatum(static_cast<float>(data[row]));
            case FLOAT8OID:
                return Float8GetDatum(data[row]);
            case NUMERICOID:
                return DirectFunctionCall1(float8_numeric, Float8GetDatum(data[row]));
            default:
                elog(ERROR, "Unsupported target type %u for DOUBLE", target_type);
                return (Datum)0;
            }
        }

        case LogicalTypeId::VARCHAR:
        case LogicalTypeId::CHAR: {
            auto* data = FlatVector::GetData<string_t>(vec);
            const auto& str = data[row];
            const char* str_data = str.GetData();
            uint32_t str_len = str.GetSize();

            switch (target_type) {
            case TEXTOID:
            case VARCHAROID:
            case BPCHAROID:
            case CHAROID:
                return PointerGetDatum(cstring_to_text_with_len(str_data, str_len));
            case JSONOID:
                return PointerGetDatum(cstring_to_text_with_len(str_data, str_len));
            case JSONBOID: {
                // Need to convert JSON string to JSONB
                // CStringGetDatum expects null-terminated string, so copy to std::string
                std::string json_str(str_data, str_len);
                return DirectFunctionCall1(jsonb_in, CStringGetDatum(json_str.c_str()));
            }
            default:
                elog(ERROR, "Unsupported target type %u for VARCHAR", target_type);
                return (Datum)0;
            }
        }

        case LogicalTypeId::BLOB: {
            auto* data = FlatVector::GetData<string_t>(vec);
            const auto& blob = data[row];
            const char* blob_data = blob.GetData();
            uint32_t blob_len = blob.GetSize();

            bytea* bytea_result = (bytea*)palloc(blob_len + VARHDRSZ);
            SET_VARSIZE(bytea_result, blob_len + VARHDRSZ);
            memcpy(VARDATA(bytea_result), blob_data, blob_len);
            return PointerGetDatum(bytea_result);
        }

        case LogicalTypeId::DATE: {
            auto* data = FlatVector::GetData<date_t>(vec);
            // DuckDB date_t is days since 1970-01-01
            // PostgreSQL DateADT is days since 2000-01-01
            int32_t duckdb_days = data[row].days;
            // Convert from Unix epoch (1970) to PostgreSQL epoch (2000)
            DateADT pg_date = duckdb_days - pg::POSTGRES_EPOCH_DAYS;
            return DateADTGetDatum(pg_date);
        }

        case LogicalTypeId::TIME: {
            auto* data = FlatVector::GetData<dtime_t>(vec);
            // DuckDB dtime_t is microseconds since midnight
            // PostgreSQL TimeADT is also microseconds since midnight
            TimeADT pg_time = data[row].micros;
            return TimeADTGetDatum(pg_time);
        }

        case LogicalTypeId::TIMESTAMP:
        case LogicalTypeId::TIMESTAMP_MS:
        case LogicalTypeId::TIMESTAMP_NS:
        case LogicalTypeId::TIMESTAMP_SEC: {
            auto* data = FlatVector::GetData<timestamp_t>(vec);
            // DuckDB timestamp_t is microseconds since Unix epoch (1970-01-01)
            // PostgreSQL Timestamp is microseconds since PostgreSQL epoch (2000-01-01)
            int64_t duckdb_micros = data[row].value;
            // Convert from Unix epoch to PostgreSQL epoch
            int64_t pg_timestamp_val = duckdb_micros - pg::TIMESTAMP_EPOCH_DIFF_US;
            return TimestampGetDatum(static_cast<::Timestamp>(pg_timestamp_val));
        }

        case LogicalTypeId::TIMESTAMP_TZ: {
            auto* data = FlatVector::GetData<timestamp_t>(vec);
            int64_t duckdb_micros = data[row].value;
            // Convert from Unix epoch to PostgreSQL epoch
            int64_t pg_timestamptz_val = duckdb_micros - pg::TIMESTAMP_EPOCH_DIFF_US;
            return TimestampTzGetDatum(static_cast<::TimestampTz>(pg_timestamptz_val));
        }

        case LogicalTypeId::UUID: {
            // UUID in DuckDB is stored as hugeint_t (INT128), not string
            // We need to convert it to string format first
            auto* data = FlatVector::GetData<hugeint_t>(vec);
            const auto& uuid_val = data[row];

            // DuckDB flips the top bit of UUIDs when storing them internally
            // to make ORDER BY consistent between UUID and VARCHAR types.
            // We must flip it back before converting to string.
            // See: duckdb/src/common/types/uuid.cpp:67 (FromString) and :80 (ToString)
            uint64_t upper_flipped = static_cast<uint64_t>(uuid_val.upper) ^ (1ULL << 63);

            // Convert hugeint to UUID string format
            // DuckDB stores UUID as a 128-bit integer
            // Format: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
            char uuid_str[37]; // 36 chars + null terminator
            snprintf(uuid_str,
                     sizeof(uuid_str),
                     "%08lx-%04lx-%04lx-%04lx-%012lx",
                     (unsigned long)((upper_flipped >> 32) & 0xFFFFFFFFUL),
                     (unsigned long)((upper_flipped >> 16) & 0xFFFFUL),
                     (unsigned long)(upper_flipped & 0xFFFFUL),
                     (unsigned long)((uuid_val.lower >> 48) & 0xFFFFUL),
                     (unsigned long)(uuid_val.lower & 0xFFFFFFFFFFFFUL));

            return DirectFunctionCall1(uuid_in, CStringGetDatum(uuid_str));
        }

        case LogicalTypeId::HUGEINT: {
            // DuckDB HUGEINT is 128-bit integer, PostgreSQL doesn't have native 128-bit
            // For simplicity, convert large values to NULL and warn
            // A more complete implementation would use NUMERIC for large values
            elog(WARNING, "HUGEINT values not fully supported, converting to NULL");
            is_null = true;
            return (Datum)0;
        }

        case LogicalTypeId::LIST: {
            // DuckDB LIST maps to PostgreSQL array - convert directly
            // Cast away const for DuckDB API
            auto& vec_mut = const_cast<Vector&>(vec);
            auto list_entry = ListVector::GetData(vec_mut)[row];
            auto& list_child = ListVector::GetEntry(vec_mut);
            auto list_size = list_entry.length;
            auto list_offset = list_entry.offset;

            // Get element type
            auto child_type = ListType::GetChildType(duckdb_type);

            // Check if this is a nested list (multi-dimensional array)
            if (child_type.id() == LogicalTypeId::LIST) {
                // Multi-dimensional array: need to flatten and construct proper PG array
                // PostgreSQL multi-dimensional arrays are flat with dimension metadata,
                // NOT arrays of arrays.

                // First, determine the base element type by walking down the type hierarchy
                LogicalType base_type = child_type;
                while (base_type.id() == LogicalTypeId::LIST) {
                    base_type = ListType::GetChildType(base_type);
                }

                // Determine PostgreSQL element type from the base type
                Oid element_type;
                switch (base_type.id()) {
                case LogicalTypeId::SMALLINT:
                    element_type = INT2OID;
                    break;
                case LogicalTypeId::INTEGER:
                    element_type = INT4OID;
                    break;
                case LogicalTypeId::BIGINT:
                    element_type = INT8OID;
                    break;
                case LogicalTypeId::FLOAT:
                    element_type = FLOAT4OID;
                    break;
                case LogicalTypeId::DOUBLE:
                    element_type = FLOAT8OID;
                    break;
                case LogicalTypeId::VARCHAR:
                    element_type = TEXTOID;
                    break;
                default:
                    elog(ERROR, "Unsupported nested LIST element type: %s", base_type.ToString().c_str());
                    return (Datum)0;
                }

                if (list_size == 0) {
                    ::ArrayType* pg_array = construct_empty_array(element_type);
                    return PointerGetDatum(pg_array);
                }

                // For 2D arrays, use a simple direct approach
                // Get dimensions by examining the structure
                // Dim 0: list_size (number of inner lists)
                // Dim 1: length of first inner list (assuming rectangular array)
                auto first_inner_entry = ListVector::GetData(list_child)[list_offset];
                int dim0 = (int)list_size;
                int dim1 = (int)first_inner_entry.length;
                int total_elements = dim0 * dim1;

                // Allocate flat arrays for all elements
                Datum* elem_datums = (Datum*)palloc(static_cast<size_t>(total_elements) * sizeof(Datum));
                bool* elem_nulls = (bool*)palloc(static_cast<size_t>(total_elements) * sizeof(bool));

                // Get the data vector (grandchild of the outer list)
                auto& data_vec = ListVector::GetEntry(list_child);

                // Flatten the 2D structure by iterating through inner lists
                int flat_idx = 0;
                for (int i = 0; i < dim0; i++) {
                    auto inner_entry = ListVector::GetData(list_child)[list_offset + i];
                    for (idx_t j = 0; j < inner_entry.length; j++) {
                        elem_datums[flat_idx] = duckdb_value_to_pg_datum(
                            data_vec, inner_entry.offset + j, element_type, -1, elem_nulls[flat_idx]);
                        flat_idx++;
                    }
                }

                // Get element type info
                int16 elem_len;
                bool elem_byval;
                char elem_align;
                get_typlenbyvalalign(element_type, &elem_len, &elem_byval, &elem_align);

                // Construct 2D PostgreSQL array
                int dims[2] = {dim0, dim1};
                int lbs[2] = {1, 1}; // 1-indexed

                ::ArrayType* pg_array = construct_md_array(elem_datums,
                                                           elem_nulls,
                                                           2, // ndims
                                                           dims,
                                                           lbs,
                                                           element_type,
                                                           elem_len,
                                                           elem_byval,
                                                           elem_align);

                pfree(elem_datums);
                pfree(elem_nulls);

                return PointerGetDatum(pg_array);
            }

            // 1D array case - original logic
            // Determine PostgreSQL element type
            Oid element_type;
            if (type_is_array(target_type)) {
                element_type = get_element_type(target_type);
            } else {
                // Fallback: infer from DuckDB child type
                switch (child_type.id()) {
                case LogicalTypeId::SMALLINT:
                    element_type = INT2OID;
                    break;
                case LogicalTypeId::INTEGER:
                    element_type = INT4OID;
                    break;
                case LogicalTypeId::BIGINT:
                    element_type = INT8OID;
                    break;
                case LogicalTypeId::FLOAT:
                    element_type = FLOAT4OID;
                    break;
                case LogicalTypeId::DOUBLE:
                    element_type = FLOAT8OID;
                    break;
                case LogicalTypeId::VARCHAR:
                    element_type = TEXTOID;
                    break;
                default:
                    elog(ERROR, "Unsupported LIST element type: %s", child_type.ToString().c_str());
                    return (Datum)0;
                }
            }

            if (list_size == 0) {
                // Empty array - disambiguate PostgreSQL ArrayType
                ::ArrayType* pg_array = construct_empty_array(element_type);
                return PointerGetDatum(pg_array);
            }

            // Convert each element
            Datum* elem_datums = (Datum*)palloc(static_cast<size_t>(list_size) * sizeof(Datum));
            bool* elem_nulls = (bool*)palloc(static_cast<size_t>(list_size) * sizeof(bool));

            for (idx_t i = 0; i < list_size; i++) {
                elem_datums[i] = duckdb_value_to_pg_datum(list_child,
                                                          list_offset + i,
                                                          element_type,
                                                          -1, // No typmod for array elements
                                                          elem_nulls[i]);
            }

            // Get element type info
            int16 elem_len;
            bool elem_byval;
            char elem_align;
            get_typlenbyvalalign(element_type, &elem_len, &elem_byval, &elem_align);

            // Construct 1D PostgreSQL array - disambiguate PostgreSQL ArrayType
            int dims[1] = {(int)list_size};
            int lbs[1] = {1}; // 1-indexed

            ::ArrayType* pg_array = construct_md_array(elem_datums,
                                                       elem_nulls,
                                                       1, // ndims
                                                       dims,
                                                       lbs,
                                                       element_type,
                                                       elem_len,
                                                       elem_byval,
                                                       elem_align);

            pfree(elem_datums);
            pfree(elem_nulls);

            return PointerGetDatum(pg_array);
        }

        default:
            elog(ERROR, "Unsupported DuckDB type: %s", duckdb_type.ToString().c_str());
            return (Datum)0;
        }
    } catch (const std::exception& e) {
        elog(ERROR, "Error converting DuckDB value to PostgreSQL Datum: %s", e.what());
        return (Datum)0;
    }
}

} // namespace pg
