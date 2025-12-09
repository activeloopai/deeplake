#include <duckdb.hpp>

#include "duckdb_deeplake_convert.hpp"
#include "utils.hpp"

#include <codecs/compression.hpp>
#include <nd/adapt.hpp>
#include <nd/none.hpp>

namespace {

using bytea_type = std::span<const uint8_t>;

auto switch_duckdb_type(const duckdb::LogicalType& duckdb_type, auto&& f)
{
    switch (duckdb_type.id()) {
    case duckdb::LogicalTypeId::BOOLEAN:
        return f.template operator()<bool>();
    case duckdb::LogicalTypeId::TINYINT:
        return f.template operator()<int8_t>();
    case duckdb::LogicalTypeId::SMALLINT:
        return f.template operator()<int16_t>();
    case duckdb::LogicalTypeId::INTEGER:
        return f.template operator()<int32_t>();
    case duckdb::LogicalTypeId::BIGINT:
    case duckdb::LogicalTypeId::HUGEINT:
        return f.template operator()<int64_t>();
    case duckdb::LogicalTypeId::UTINYINT:
        return f.template operator()<uint8_t>();
    case duckdb::LogicalTypeId::USMALLINT:
        return f.template operator()<uint16_t>();
    case duckdb::LogicalTypeId::UINTEGER:
        return f.template operator()<uint32_t>();
    case duckdb::LogicalTypeId::UBIGINT:
        return f.template operator()<uint64_t>();
    case duckdb::LogicalTypeId::FLOAT:
        return f.template operator()<float>();
    case duckdb::LogicalTypeId::DOUBLE:
        return f.template operator()<double>();
    case duckdb::LogicalTypeId::VARCHAR: {
        if (duckdb_type.IsJSONType()) {
            return f.template operator()<nd::dict>();
        }
        return f.template operator()<std::string>();
    }
    case duckdb::LogicalTypeId::CHAR:
    case duckdb::LogicalTypeId::UUID:
        return f.template operator()<std::string>();
    case duckdb::LogicalTypeId::BLOB:
        return f.template operator()<bytea_type>();
    case duckdb::LogicalTypeId::DATE:
        return f.template operator()<int32_t>();
    case duckdb::LogicalTypeId::TIME:
        return f.template operator()<int64_t>();
    case duckdb::LogicalTypeId::TIMESTAMP:
    case duckdb::LogicalTypeId::TIMESTAMP_MS:
    case duckdb::LogicalTypeId::TIMESTAMP_NS:
    case duckdb::LogicalTypeId::TIMESTAMP_SEC:
    case duckdb::LogicalTypeId::TIMESTAMP_TZ:
        return f.template operator()<int64_t>();
    default:
        throw duckdb::NotImplementedException("Unsupported DuckDB type: " + duckdb_type.ToString());
    }
}

template <typename T>
T to_cpp_value(const duckdb::Value& val)
{
    ASSERT(!val.IsNull());
    if constexpr (std::is_same_v<T, std::string>) {
        return val.ToString();
    } else if constexpr (std::is_same_v<T, int32_t>) {
        return val.GetValue<int32_t>();
    } else if constexpr (std::is_same_v<T, int64_t>) {
        return val.GetValue<int64_t>();
    } else if constexpr (std::is_same_v<T, bytea_type>) {
        // For bytea/blob types, get the string_t and convert to span
        // Use GetValueUnsafe since GetValue<string_t> template may not be instantiated
        auto str_val = val.GetValueUnsafe<duckdb::string_t>();
        return bytea_type(reinterpret_cast<const uint8_t*>(str_val.GetData()), str_val.GetSize());
    } else {
        return val.GetValue<T>();
    }
}

nd::array to_deeplake_value_as_array_list(const duckdb::vector<duckdb::Value>& values)
{
    std::vector<nd::array> arr;
    arr.reserve(values.size());
    for (const auto& v : values) {
        arr.push_back(pg::to_deeplake_value(v));
    }
    return nd::dynamic(std::move(arr));
}

nd::array to_deeplake_value(const duckdb::LogicalType& duckdb_type, const duckdb::vector<duckdb::Value>& values)
{
    if (duckdb_type.id() == duckdb::LogicalTypeId::LIST) {
        return to_deeplake_value_as_array_list(values);
    }
    for (const auto& v : values) {
        if (v.IsNull()) {
            return to_deeplake_value_as_array_list(values);
        }
    }
    return switch_duckdb_type(duckdb_type, [&values]<typename T>() {
        if constexpr (std::is_same_v<T, bytea_type>) {
            std::vector<nd::array> arr;
            arr.reserve(values.size());
            for (const auto& val : values) {
                duckdb::string_t blob_data = duckdb::StringValue::Get(val);
                const uint8_t* data = reinterpret_cast<const uint8_t*>(blob_data.GetData());
                size_t len = blob_data.GetSize();
                arr.emplace_back(nd::adapt(base::memory_buffer(std::vector<uint8_t>(data, data + len)), nd::dtype::byte));
            }
            return nd::dynamic(std::move(arr));
        }
        using U = std::conditional_t<std::is_same_v<T, nd::dict>, std::string, T>;
        std::vector<std::conditional_t<std::is_same_v<U, bool>, uint8_t, U>> arr;
        arr.reserve(values.size());
        for (const auto& val : values) {
            if constexpr (std::is_same_v<T, nd::dict>) {
                arr.push_back(to_cpp_value<std::string>(val));
            } else {
                arr.push_back(to_cpp_value<T>(val));
            }
        }
        return nd::adapt(std::move(arr));
    });
}

nd::dtype to_deeplake_nd_dtype(const duckdb::LogicalType& type)
{
    return switch_duckdb_type(type, []<typename T>() {
        return nd::dtype_enum<T>::value;
    });
}

} // unnamed namespace

namespace pg {

deeplake_core::type to_deeplake_type(const duckdb::LogicalType& duckdb_type)
{
    deeplake_core::type result;
    switch (duckdb_type.id()) {
    case duckdb::LogicalTypeId::VARCHAR: {
        if (duckdb_type.IsJSONType()) {
            result = deeplake_core::type::dict();
            break;
        }
        result = deeplake_core::type::text(codecs::compression::null);
        break;
    }
    case duckdb::LogicalTypeId::CHAR:
    case duckdb::LogicalTypeId::UUID:
        result = deeplake_core::type::text(codecs::compression::null);
        break;
    case duckdb::LogicalTypeId::BLOB: {
        result = deeplake_core::type::generic(nd::type::array(nd::dtype::byte));
        break;
    }
    case duckdb::LogicalTypeId::LIST: {
        // Array/LIST type - determine element type
        auto child_type = duckdb::ListType::GetChildType(duckdb_type);
        if (child_type.id() == duckdb::LogicalTypeId::LIST) {
            return to_deeplake_type(child_type);
        }
        auto element_dtype = to_deeplake_nd_dtype(child_type.id());

        if (nd::dtype_is_numeric(element_dtype)) {
            // Numeric array - use generic type with array dtype
            result = deeplake_core::type::generic(nd::type::array(element_dtype));
        } else {
            // String array - use generic type with array of bytes (for variable length strings)
            result = deeplake_core::type::generic(nd::type::array(nd::dtype::byte));
        }
        break;
    }
    default: {
        const auto element_dtype = to_deeplake_nd_dtype(duckdb_type.id());
        result = deeplake_core::type::generic(nd::type::scalar(element_dtype));
        break;
    }
    }
    return result;
}

nd::array to_deeplake_value(const duckdb::Value& value)
{
    const auto& duckdb_type = value.type();
    if (duckdb_type.id() == duckdb::LogicalTypeId::LIST) {
        if (value.IsNull()) {
            const auto dt = to_deeplake_nd_dtype(duckdb_type.id());
            return nd::none(dt, 1);
        }
        const auto& list_value = duckdb::ListValue::GetChildren(value);
        const auto child_type = duckdb::ListType::GetChildType(duckdb_type);
        return ::to_deeplake_value(child_type, list_value);
    }
    return switch_duckdb_type(duckdb_type, [&value]<typename T>() {
        if (value.IsNull()) {
            if constexpr (std::is_same_v<T, bool>) {
                return nd::adapt(uint8_t(0));
            } else {
                constexpr auto dt = nd::dtype_enum<T>::value;
                return (nd::dtype_is_numeric(dt) ? nd::adapt(T()) : nd::none(dt, 1));
            }
        }
        if constexpr (std::is_same_v<T, nd::dict>) {
            auto json_str = to_cpp_value<std::string>(value);
            return nd::adapt(json_str);
        } else if constexpr (std::is_same_v<T, bytea_type>) {
            duckdb::string_t blob_data = value.GetValueUnsafe<duckdb::string_t>();
            const uint8_t* data = reinterpret_cast<const uint8_t*>(blob_data.GetData());
            size_t len = blob_data.GetSize();
            return nd::adapt(base::memory_buffer(std::vector<uint8_t>(data, data + len)), nd::dtype::byte);
        } else {
            return nd::adapt(to_cpp_value<T>(value));
        }
    });
}

nd::array to_deeplake_value(std::shared_ptr<duckdb::Vector>&& vector, size_t total_rows)
{
    const auto& duckdb_type = vector->GetType();
    ASSERT(duckdb_type.IsNumeric() && duckdb::GetTypeIdSize(duckdb_type.InternalType()) <= 8);

    // For arithmetic types, use zero-copy by wrapping DuckDB's internal data pointer
    return switch_duckdb_type(duckdb_type, [vector = std::move(vector), total_rows]<typename T>() -> nd::array {
        // Get pointer to DuckDB's internal data
        auto* src_data = duckdb::FlatVector::GetData<T>(*vector);
        const auto dt = nd::dtype_enum<T>::value;
        auto data_span = std::span<const uint8_t>(
            reinterpret_cast<const uint8_t*>(src_data),
            total_rows * sizeof(T)
        );
        return nd::array(nd::impl::std_span_array_nd(std::move(vector), data_span,
                         icm::shape{static_cast<int64_t>(total_rows)}, dt));
    });
}

} // namespace pg
