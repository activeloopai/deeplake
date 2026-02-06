#pragma once

#include "exceptions.hpp"

#include <icm/shape.hpp>
#include <icm/vector.hpp>
#include <nd/adapt.hpp>
#include <nd/array.hpp>
#include <nd/none.hpp>

#ifdef __cplusplus
extern "C" {
#endif

#include <utils/array.h>
#include <utils/date.h>
#include <utils/fmgroids.h>
#include <utils/jsonb.h>
#include <utils/numeric.h>
#include <utils/timestamp.h>
#include <utils/uuid.h>

#ifdef __cplusplus
} /// extern "C"
#endif

#include <algorithm>

namespace pg {

using array_type = nd::array;

} // namespace pg

namespace pg::utils {

template <typename T>
class pointer_array
{
public:
    struct holder_t
    {
        explicit holder_t(T* d, std::size_t size)
            : data_(d)
            , shape_(size)
        {
        }

        explicit holder_t(T* d, std::size_t size, std::size_t size2)
            : data_(d)
            , shape_({size, size2})
        {
        }

        T* data_;
        icm::shape shape_;
    };

public:
    explicit pointer_array(T* value, std::size_t size)
        : value_(std::make_shared<holder_t>(value, size))
    {
    }

    explicit pointer_array(T* value, std::size_t size, std::size_t size2)
        : value_(std::make_shared<holder_t>(value, size, size2))
    {
    }

    enum nd::dtype dtype() const
    {
        return nd::dtype_enum_v<T>;
    }

    std::span<const uint8_t> data() const
    {
        auto len =
            value_->shape_.size() == 1 ? value_->shape_.front() : (value_->shape_.front() * value_->shape_.back());
        return base::span_cast<const uint8_t>(std::span<const T>(value_->data_, value_->data_ + len));
    }

    const auto& owner() const
    {
        return value_;
    }

    icm::shape shape() const
    {
        return value_->shape_;
    }

    constexpr bool is_dynamic() const noexcept
    {
        return false;
    }

private:
    const std::shared_ptr<holder_t> value_;
};

/// Convert PostgreSQL array to nd array (template version for all numeric types)
template <typename T>
inline pg::array_type pg_to_nd_typed(ArrayType* array, bool copy_data = true)
{
    if (array == nullptr) {
        throw exception("Null array not allowed");
    }
    const auto ndim = ARR_NDIM(array);
    if (ndim < 1 || ndim > 255) {
        throw exception(fmt::format("Array dimensions must be between 1 and 255, provided: {}", ndim));
    }

    auto* data = reinterpret_cast<T*>(ARR_DATA_PTR(array));
    const auto* dims = ARR_DIMS(array);
    const auto nelems = ArrayGetNItems(ndim, dims);

    if (ndim == 1) {
        // 1D array - simple case
        if (copy_data) {
            return nd::adapt(std::vector<T>(data, data + nelems));
        }
        return pg::array_type(pointer_array(data, static_cast<std::size_t>(nelems)));
    } else if (ndim == 2) {
        // 2D array - optimized path
        const auto nrows = dims[0];
        const auto ncols = dims[1];
        if (copy_data) {
            icm::vector<nd::array> data_vector;
            data_vector.reserve(static_cast<size_t>(nrows));
            for (int i = 0; i < nrows; ++i) {
                data_vector.emplace_back(nd::adapt(std::vector<T>(data + static_cast<size_t>(i) * static_cast<size_t>(ncols),
                                                                   data + static_cast<size_t>(i + 1) * static_cast<size_t>(ncols))));
            }
            return nd::dynamic(data_vector);
        }
        return pg::array_type(pointer_array(data, static_cast<std::size_t>(nrows), static_cast<std::size_t>(ncols)));
    } else {
        // N-dimensional array (N > 2)
        // Build icm::shape directly from PostgreSQL dimensions array
        icm::shape shape(dims, dims + ndim);

        if (copy_data) {
            // Copy data into vector and create nd::array with shape using nd::adapt
            std::vector<T> vec_data(data, data + nelems);
            return nd::adapt(std::move(vec_data), std::move(shape));
        } else {
            // Use pointer_array wrapper - note: only supports 1D and 2D currently
            throw exception(fmt::format("Non-copy mode only supports 1D and 2D arrays, requested {}D", ndim));
        }
    }
}

/// Convert PostgreSQL array to nd array (dispatcher based on element type)
inline pg::array_type pg_to_nd(ArrayType* array, bool copy_data = true)
{
    if (array == nullptr) {
        throw exception("Null array not allowed");
    }

    Oid elem_type = ARR_ELEMTYPE(array);
    switch (elem_type) {
    case INT2OID:
        return pg_to_nd_typed<int16_t>(array, copy_data);
    case INT4OID:
        return pg_to_nd_typed<int32_t>(array, copy_data);
    case INT8OID:
        return pg_to_nd_typed<int64_t>(array, copy_data);
    case FLOAT4OID:
        return pg_to_nd_typed<float>(array, copy_data);
    case FLOAT8OID:
        return pg_to_nd_typed<double>(array, copy_data);
    default:
        throw exception(fmt::format("Unsupported array element type: {}", elem_type));
    }
}

/// Convert nd::array to PostgreSQL array (template version for all numeric types)
template <typename T, Oid ElementOID>
inline Datum nd_to_pg_typed(const pg::array_type& arr)
{
    // Check dimensionality
    auto shape = arr.shape();
    const auto ndim = shape.size();

    if (ndim < 1 || ndim > 255) {
        throw exception(fmt::format("Array dimensions must be between 1 and 255, provided: {}", ndim));
    }

    // Get element properties based on type
    int16_t elem_size = 0;
    bool elem_byval = false;
    char elem_align = 'i';

    if constexpr (std::is_same_v<T, int16_t>) {
        elem_size = sizeof(int16_t);
        elem_byval = true;
        elem_align = 's';
    } else if constexpr (std::is_same_v<T, int32_t>) {
        elem_size = sizeof(int32_t);
        elem_byval = true;
        elem_align = 'i';
    } else if constexpr (std::is_same_v<T, int64_t>) {
        elem_size = sizeof(int64_t);
        elem_byval = true;
        elem_align = 'd';
    } else if constexpr (std::is_same_v<T, float>) {
        elem_size = sizeof(float);
        elem_byval = true;
        elem_align = 'f';
    } else if constexpr (std::is_same_v<T, double>) {
        elem_size = sizeof(double);
        elem_byval = true;
        elem_align = 'd';
    } else {
        throw exception("Unsupported numeric type for array conversion");
    }

    // Helper lambda to convert a single element to Datum
    auto to_datum = [](const nd::array& val) -> Datum {
        if constexpr (std::is_same_v<T, int16_t>) {
            return Int16GetDatum(val.template value<int16_t>(0));
        } else if constexpr (std::is_same_v<T, int32_t>) {
            return Int32GetDatum(val.template value<int32_t>(0));
        } else if constexpr (std::is_same_v<T, int64_t>) {
            return Int64GetDatum(val.template value<int64_t>(0));
        } else if constexpr (std::is_same_v<T, float>) {
            return Float4GetDatum(val.template value<float>(0));
        } else if constexpr (std::is_same_v<T, double>) {
            return Float8GetDatum(val.template value<double>(0));
        }
    };

    // Calculate total number of elements
    size_t total_elements = 1;
    for (size_t i = 0; i < ndim; ++i) {
        total_elements *= static_cast<size_t>(shape[i]);
    }

    // Allocate memory for all elements
    Datum* elements = (Datum*)palloc(static_cast<size_t>(total_elements) * sizeof(Datum));

    // Flatten the nd::array into the elements array
    // Use recursive approach to handle arbitrary dimensions
    std::function<void(const nd::array&, size_t&)> flatten;
    flatten = [&](const nd::array& sub_arr, size_t& idx) {
        if (sub_arr.shape().size() == 1) {
            // Base case: 1D array, copy all elements
            for (size_t i = 0; i < sub_arr.size(); ++i) {
                elements[idx++] = to_datum(sub_arr[i]);
            }
        } else {
            // Recursive case: iterate through first dimension
            for (size_t i = 0; i < sub_arr.size(); ++i) {
                flatten(sub_arr[i], idx);
            }
        }
    };

    size_t idx = 0;
    if (ndim == 1) {
        // Optimized path for 1D arrays
        for (size_t i = 0; i < arr.size(); ++i) {
            elements[i] = to_datum(arr[static_cast<size_t>(i)]);
        }
    } else {
        // General path for N-D arrays
        flatten(arr, idx);
    }

    // Build dimensions and lower bounds arrays
    std::vector<int32_t> dimensions(ndim);
    std::vector<int32_t> lower_bounds(ndim, 1); // PostgreSQL arrays default to 1-based indexing

    for (size_t i = 0; i < ndim; ++i) {
        dimensions[i] = static_cast<int32_t>(shape[i]);
    }

    // Create PostgreSQL multidimensional array
    ArrayType* pg_array = construct_md_array(elements,                        // Values
                                             nullptr,                         // Nulls (no nulls)
                                             static_cast<int>(ndim),          // Number of dimensions
                                             dimensions.data(),               // Dimensions array
                                             lower_bounds.data(),             // Lower bounds (all 1s)
                                             ElementOID,                      // Element OID type
                                             elem_size,                       // Element width
                                             elem_byval,                      // Pass by value?
                                             elem_align                       // Element alignment
    );

    return PointerGetDatum(pg_array);
}

/// Convert nd::array to PostgreSQL array (defaults to FLOAT4ARRAYOID for backward compatibility)
inline Datum nd_to_pg(const pg::array_type& arr)
{
    return nd_to_pg_typed<float, FLOAT4OID>(arr);
}

/// Convert nd::array to PostgreSQL array with specified target type
inline Datum nd_to_pg(const pg::array_type& arr, Oid target_array_type)
{
    switch (target_array_type) {
    case INT2ARRAYOID:
        return nd_to_pg_typed<int16_t, INT2OID>(arr);
    case INT4ARRAYOID:
        return nd_to_pg_typed<int32_t, INT4OID>(arr);
    case INT8ARRAYOID:
        return nd_to_pg_typed<int64_t, INT8OID>(arr);
    case FLOAT4ARRAYOID:
        return nd_to_pg_typed<float, FLOAT4OID>(arr);
    case FLOAT8ARRAYOID:
        return nd_to_pg_typed<double, FLOAT8OID>(arr);
    default:
        throw exception(fmt::format("Unsupported target array type: {}", target_array_type));
    }
}

/// Convert deeplake_core::type to PostgreSQL type string
/// TODO: Review / revisit
inline std::string nd_to_pg_type(deeplake_core::type t)
{
    auto kind = t.kind();
    std::string res;
    switch (kind) {
    case deeplake_core::type_kind::generic:
        res = t.to_string();
        break;
    case deeplake_core::type_kind::link: {
        // Check if link contains bytes (file type) vs other types
        auto inner = t.as_link().get_type();
        if (inner->is_generic() && inner->data_type().get_dtype() == nd::dtype::byte) {
            res = "file";
        } else {
            res = "text";
        }
        break;
    }
    case deeplake_core::type_kind::text:
        res = "text";
        break;
    case deeplake_core::type_kind::dict:
    case deeplake_core::type_kind::struct_:
        res = "json";
        break;
    case deeplake_core::type_kind::embedding:
        res = "float4";
        break;
    case deeplake_core::type_kind::image:
        res = "image";
        break;
    case deeplake_core::type_kind::video:
        res = "video";
        break;
    case deeplake_core::type_kind::audio:
    case deeplake_core::type_kind::bmask:
    case deeplake_core::type_kind::smask:
    case deeplake_core::type_kind::medical:
        res = "bytea";
        break;
    case deeplake_core::type_kind::polygon:
        res = "polygon";
        break;
    case deeplake_core::type_kind::point:
        res = "point";
        break;
    case deeplake_core::type_kind::class_label:
        res = "enum";
        break;
    case deeplake_core::type_kind::bbox:
        res = "box";
        break;
    case deeplake_core::type_kind::sequence:
        res = t.as_sequence().data_type().to_string();
        break;
    default:
        throw exception(fmt::format("Unsupported type kind for PostgreSQL conversion: {}",
                                    deeplake_core::type_kind_to_string(kind)));
    }
    if (res.find("int8") != std::string::npos || res.find("int16") != std::string::npos) {
        res = "int2"; // PostgreSQL does not have 1 byte int, use int2 instead
    } else if (res.find("int32") != std::string::npos) {
        res = "int4";
    } else if (res.find("int64") != std::string::npos) {
        res = "int8";
    } else if (res.find("float16") != std::string::npos || res.find("float32") != std::string::npos) {
        res = "float4"; // PostgreSQL does not have 2 byte float, use float4 instead
    } else if (res.find("float64") != std::string::npos) {
        res = "float8";
    }
    if (t.data_type().is_array()) {
        res += "[]";
    }
    return res;
}

inline nd::array datum_to_nd(Datum value, Oid attr_typeid, int32_t typmod)
{
    switch (attr_typeid) {
    case BOOLOID: {
        return nd::adapt(DatumGetBool(value));
    }
    case INT2OID: {
        return nd::adapt(DatumGetInt16(value));
    }
    case INT4OID: {
        return nd::adapt(DatumGetInt32(value));
    }
    case INT8OID: {
        return nd::adapt(DatumGetInt64(value));
    }
    case FLOAT4OID: {
        return nd::adapt(DatumGetFloat4(value));
    }
    case FLOAT8OID: {
        return nd::adapt(DatumGetFloat8(value));
    }
    case NUMERICOID: {
        if (value == 0) {
            return nd::none(nd::dtype::float64, 0);
        }

        // First get the numeric value
        Numeric num = DatumGetNumeric(value);
        if (num == nullptr) {
            return nd::none(nd::dtype::float64, 0);
        }

        PG_TRY();
        {
            // Convert numeric to double using PostgreSQL's built-in function
            double d = DatumGetFloat8(DirectFunctionCall1(numeric_float8, NumericGetDatum(num)));
            // Check for NaN or infinity
            if (std::isnan(d) || std::isinf(d)) {
                return nd::none(nd::dtype::float64, 0);
            }
            return nd::adapt(d);
        }
        PG_CATCH();
        {
            // Handle any conversion errors
            return nd::none(nd::dtype::float64, 0);
        }
        PG_END_TRY();
    }
    case DATEOID: {
        DateADT date = DatumGetDateADT(value);
        // Store in Unix epoch (1970) instead of PostgreSQL epoch (2000)
        return nd::adapt(static_cast<int32_t>(date + pg::POSTGRES_EPOCH_DAYS));
    }
    case TIMEOID: {
        TimeADT time = DatumGetTimeADT(value);
        return nd::adapt(static_cast<int64_t>(time));
    }
    case TIMESTAMPOID: {
        Timestamp timestamp = DatumGetTimestamp(value);
        // Store in Unix epoch (1970) instead of PostgreSQL epoch (2000)
        return nd::adapt(static_cast<int64_t>(timestamp + pg::TIMESTAMP_EPOCH_DIFF_US));
    }
    case TIMESTAMPTZOID: {
        TimestampTz timestamp_tz = DatumGetTimestampTz(value);
        // Store in Unix epoch (1970) instead of PostgreSQL epoch (2000)
        return nd::adapt(static_cast<int64_t>(timestamp_tz + pg::TIMESTAMP_EPOCH_DIFF_US));
    }
    case INT2ARRAYOID: {
        ArrayType* arr = DatumGetArrayTypeP(value);
        if (arr == nullptr) {
            return nd::none(nd::dtype::int16, 1);
        }
        return pg::utils::pg_to_nd(arr);
    }
    case INT4ARRAYOID: {
        ArrayType* arr = DatumGetArrayTypeP(value);
        if (arr == nullptr) {
            return nd::none(nd::dtype::int32, 1);
        }
        return pg::utils::pg_to_nd(arr);
    }
    case INT8ARRAYOID: {
        ArrayType* arr = DatumGetArrayTypeP(value);
        if (arr == nullptr) {
            return nd::none(nd::dtype::int64, 1);
        }
        return pg::utils::pg_to_nd(arr);
    }
    case FLOAT4ARRAYOID: {
        ArrayType* arr = DatumGetArrayTypeP(value);
        if (arr == nullptr) {
            return nd::none(nd::dtype::float32, 1);
        }
        return pg::utils::pg_to_nd(arr);
    }
    case FLOAT8ARRAYOID: {
        ArrayType* arr = DatumGetArrayTypeP(value);
        if (arr == nullptr) {
            return nd::none(nd::dtype::float64, 1);
        }
        return pg::utils::pg_to_nd(arr);
    }
    case UUIDOID: {
        pg_uuid_t* uuid = DatumGetUUIDP(value);
        constexpr size_t buf_size = 36; // UUID is 16 bytes, 32 hex chars + 4 hyphens
        char buf[buf_size + 1];
        buf[buf_size] = '\0';
        snprintf(buf,
                 buf_size + 1,
                 "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
                 uuid->data[0],
                 uuid->data[1],
                 uuid->data[2],
                 uuid->data[3],
                 uuid->data[4],
                 uuid->data[5],
                 uuid->data[6],
                 uuid->data[7],
                 uuid->data[8],
                 uuid->data[9],
                 uuid->data[10],
                 uuid->data[11],
                 uuid->data[12],
                 uuid->data[13],
                 uuid->data[14],
                 uuid->data[15]);
        return nd::adapt(std::string(buf, buf_size));
    }
    case CHAROID:
    case BPCHAROID:
    case VARCHAROID: {
        if (typmod == VARHDRSZ + 1) {
            text* txt = DatumGetTextP(value);
            if (txt == nullptr) {
                return nd::none(nd::dtype::string, 0);
            }
            std::string str(VARDATA(txt), VARSIZE(txt) - VARHDRSZ);
            return nd::adapt(static_cast<int8_t>(str[0]));
        }
    }
    case TEXTOID:
    case JSONOID: {
        text* txt = DatumGetTextP(value);
        if (txt == nullptr) {
            return nd::none(nd::dtype::string, 0);
        }
        std::string str(VARDATA(txt), VARSIZE(txt) - VARHDRSZ);
        return nd::adapt(str);
    }
    case JSONBOID: {
        Jsonb* jb = DatumGetJsonbP(value);
        const char* cstr = JsonbToCString(NULL, &jb->root, VARSIZE_ANY_EXHDR(jb));
        std::string json_str(cstr);
        return nd::adapt(json_str);
    }
    case BYTEAOID: {
        bytea* b = DatumGetByteaP(value);
        if (b == nullptr) {
            return nd::none(nd::dtype::byte, 0);
        } else {
            size_t len = VARSIZE(b) - VARHDRSZ;
            return nd::adapt(base::memory_buffer(std::string(VARDATA(b), len)), nd::dtype::byte);
        }
        break;
    }
    case BYTEAARRAYOID: {
        ArrayType* arr = DatumGetArrayTypeP(value);
        if (arr == nullptr) {
            return nd::none(nd::dtype::byte, 0);
        } else {
            int nelems = ArrayGetNItems(ARR_NDIM(arr), ARR_DIMS(arr));
            icm::vector<nd::array> elements;
            elements.reserve(static_cast<size_t>(nelems));

            Datum* datums = nullptr;
            bool* nulls = nullptr;
            int count = 0;
            deconstruct_array(arr, BYTEAOID, -1, false, 'i', &datums, &nulls, &count);

            for (int i = 0; i < count; ++i) {
                if (nulls && nulls[i]) {
                    elements.push_back(nd::none(nd::dtype::byte, 0));
                } else {
                    elements.push_back(datum_to_nd(datums[i], BYTEAOID, -1));
                }
            }
            return nd::dynamic(elements);
        }
        break;
    }
    case VARCHARARRAYOID:
    case TEXTARRAYOID: {
        ArrayType* arr = DatumGetArrayTypeP(value);
        if (arr == nullptr) {
            return nd::none(nd::dtype::string, 1);
        } else {
            int nelems = ArrayGetNItems(ARR_NDIM(arr), ARR_DIMS(arr));
            icm::vector<nd::array> elements;
            elements.reserve(static_cast<size_t>(nelems));
            Datum* datums = nullptr;
            bool* nulls = nullptr;
            int32_t count = 0;
            auto tid = (attr_typeid == TEXTARRAYOID) ? TEXTOID : VARCHAROID;
            deconstruct_array(arr, tid, -1, false, 'i', &datums, &nulls, &count);

            for (int32_t i = 0; i < count; ++i) {
                if (nulls && nulls[i]) {
                    elements.push_back(nd::none(nd::dtype::string, 1));
                } else {
                    elements.push_back(datum_to_nd(datums[i], tid, -1));
                }
            }
            return nd::dynamic(elements);
        }
        break;
    }
    default:
        break;
    }
    elog(ERROR, "Unsupported type for conversion to nd::array: %u", attr_typeid);
    return nd::none(nd::dtype::unknown, 0);
}

// Helper function to convert nd::array to PostgreSQL Datum based on attribute type
inline Datum nd_to_datum(const nd::array& curr_val, Oid attr_typeid, int32_t typmod)
{
    switch (attr_typeid) {
    case BOOLOID: {
        return BoolGetDatum(curr_val.value<bool>(0));
    }
    case INT2OID: {
        return Int16GetDatum(curr_val.value<int16>(0));
    }
    case INT4OID: {
        return Int32GetDatum(curr_val.value<int32>(0));
    }
    case INT8OID: {
        return Int64GetDatum(curr_val.value<int64>(0));
    }
    case FLOAT4OID: {
        return Float4GetDatum(curr_val.value<float>(0));
    }
    case FLOAT8OID: {
        return Float8GetDatum(curr_val.value<double>(0));
    }
    case NUMERICOID: {
        const auto val = curr_val.value<double>(0);
        return DirectFunctionCall1(float8_numeric, Float8GetDatum(val));
    }
    case DATEOID: {
        int32_t date = curr_val.value<int32_t>(0);
        // Convert from Unix epoch (1970) back to PostgreSQL epoch (2000)
        return DateADTGetDatum(date - pg::POSTGRES_EPOCH_DAYS);
    }
    case TIMEOID: {
        int64_t time = curr_val.value<int64>(0);
        return TimeADTGetDatum(static_cast<TimeADT>(time));
    }
    case TIMESTAMPOID: {
        int64_t timestamp = curr_val.value<int64_t>(0);
        // Convert from Unix epoch (1970) back to PostgreSQL epoch (2000)
        return TimestampGetDatum(static_cast<Timestamp>(timestamp - pg::TIMESTAMP_EPOCH_DIFF_US));
    }
    case TIMESTAMPTZOID: {
        int64_t timestamp_tz = curr_val.value<int64_t>(0);
        // Convert from Unix epoch (1970) back to PostgreSQL epoch (2000)
        return TimestampTzGetDatum(static_cast<TimestampTz>(timestamp_tz - pg::TIMESTAMP_EPOCH_DIFF_US));
    }
    case INT2ARRAYOID: {
        return pg::utils::nd_to_pg(curr_val, INT2ARRAYOID);
    }
    case INT4ARRAYOID: {
        return pg::utils::nd_to_pg(curr_val, INT4ARRAYOID);
    }
    case INT8ARRAYOID: {
        return pg::utils::nd_to_pg(curr_val, INT8ARRAYOID);
    }
    case FLOAT4ARRAYOID: {
        return pg::utils::nd_to_pg(curr_val, FLOAT4ARRAYOID);
    }
    case FLOAT8ARRAYOID: {
        return pg::utils::nd_to_pg(curr_val, FLOAT8ARRAYOID);
    }
    case UUIDOID: {
        auto str = base::string_view_cast(base::span_cast<const char>(curr_val.data()));
        // Treat empty string as NULL for UUID columns (same as duckdb_deeplake_scan.cpp)
        if (str.empty()) {
            // Return NULL datum - caller must set is_null flag appropriately
            return (Datum)0;
        }
        // CStringGetDatum expects a null-terminated string, hence copy the string
        std::string str_copy(str.data(), str.size());
        Datum uuid = DirectFunctionCall1(uuid_in, CStringGetDatum(str_copy.c_str()));
        return uuid;
    }
    case CHAROID:
    case VARCHAROID:
    case TEXTOID:
    case BPCHAROID: {
        auto str = base::string_view_cast(base::span_cast<const char>(curr_val.data()));
        return PointerGetDatum(cstring_to_text_with_len(str.data(), static_cast<int>(str.size())));
    }
    case JSONOID: {
        if (curr_val.is_none()) {
            return (Datum)0;
        }
        std::string json_str;
        try {
            if (curr_val.dtype() == nd::dtype::string) {
                json_str = base::string_view_cast(base::span_cast<const char>(curr_val.data()));
            } else if (curr_val.dtype() == nd::dtype::object) {
                json_str = curr_val.dict_value(0).serialize();
            } else {
                throw pg::exception(
                    fmt::format("Cannot convert nd::array with dtype {} to JSON", nd::dtype_to_str(curr_val.dtype())));
            }
        } catch (const std::exception& e) {
            elog(WARNING, "Error dumping JSON: %s", e.what());
            return (Datum)0;
        }
        return PointerGetDatum(cstring_to_text_with_len(json_str.data(), static_cast<int>(json_str.size())));
    }
    case JSONBOID: {
        if (curr_val.is_none()) {
            return (Datum)0;
        }
        std::string json_str;
        try {
            if (curr_val.dtype() == nd::dtype::string) {
                json_str = base::string_view_cast(base::span_cast<const char>(curr_val.data()));
            } else if (curr_val.dtype() == nd::dtype::object) {
                json_str = curr_val.dict_value(0).serialize();
            } else {
                throw pg::exception(
                    fmt::format("Cannot convert nd::array with dtype {} to JSON", nd::dtype_to_str(curr_val.dtype())));
            }
        } catch (const std::exception& e) {
            elog(WARNING, "Error dumping JSONB: %s", e.what());
            return (Datum)0;
        }
        return DirectFunctionCall1(jsonb_in, CStringGetDatum(json_str.c_str()));
    }
    case BYTEAOID: {
        if (curr_val.is_none()) {
            return (Datum)0;
        }
        std::span<const uint8_t> bytea_data = curr_val.data();
        size_t bytea_size = bytea_data.size();
        if (bytea_size == 0) {
            return (Datum)0;
        }
        bytea* bytea_ptr = (bytea*)palloc(bytea_size + static_cast<size_t>(VARHDRSZ));
        SET_VARSIZE(bytea_ptr, static_cast<int32_t>(bytea_size + static_cast<size_t>(VARHDRSZ)));
        memcpy(VARDATA(bytea_ptr), bytea_data.data(), bytea_size);
        return PointerGetDatum(bytea_ptr);
    }
    case BYTEAARRAYOID: {
        auto shape = curr_val.shape();
        size_t num_elements = static_cast<size_t>(shape[0]);
        Datum* elements = (Datum*)palloc(num_elements * sizeof(Datum));
        bool* nulls = (bool*)palloc(num_elements * sizeof(bool));
        bool has_nulls = false;

        for (size_t i = 0; i < num_elements; ++i) {
            auto element = curr_val[static_cast<size_t>(i)];
            if (element.is_none()) {
                elements[i] = (Datum)0;
                nulls[i] = true;
                has_nulls = true;
            } else {
                // Convert each element to bytea
                std::span<const uint8_t> bytea_data = element.data();
                size_t bytea_size = bytea_data.size();
                bytea* bytea_ptr = (bytea*)palloc(bytea_size + static_cast<size_t>(VARHDRSZ));
                SET_VARSIZE(bytea_ptr, static_cast<int32_t>(bytea_size + static_cast<size_t>(VARHDRSZ)));
                memcpy(VARDATA(bytea_ptr), bytea_data.data(), bytea_size);
                elements[i] = PointerGetDatum(bytea_ptr);
                nulls[i] = false;
            }
        }

        // Create array with proper null bitmap support
        int dims[1] = {static_cast<int>(num_elements)};
        int lbs[1] = {1}; // Lower bound

        ArrayType* pg_array = construct_md_array(elements,                    // Values
                                                 has_nulls ? nulls : nullptr, // Null bitmap
                                                 1,                           // Number of dimensions
                                                 dims,                        // Dimensions array
                                                 lbs,                         // Lower bounds
                                                 BYTEAOID,                    // Element OID type
                                                 -1,                          // Element size (-1 for variable length)
                                                 false,                       // Pass by value? (false for bytea)
                                                 'i'                          // Element alignment
        );
        return PointerGetDatum(pg_array);
    }
    case VARCHARARRAYOID:
    case TEXTARRAYOID: {
        auto shape = curr_val.shape();
        size_t num_elements = static_cast<size_t>(shape[0]);
        Datum* elements = (Datum*)palloc(num_elements * sizeof(Datum));
        bool* nulls = (bool*)palloc(num_elements * sizeof(bool));
        bool has_nulls = false;
        Oid element_typeid = (attr_typeid == TEXTARRAYOID) ? TEXTOID : VARCHAROID;
        for (size_t i = 0; i < num_elements; ++i) {
            auto element = curr_val[static_cast<size_t>(i)];
            if (element.is_none()) {
                elements[i] = (Datum)0;
                nulls[i] = true;
                has_nulls = true;
            } else {
                // Convert each element to text
                auto str = base::string_view_cast(base::span_cast<const char>(element.data()));
                elements[i] = PointerGetDatum(cstring_to_text_with_len(str.data(), static_cast<int>(str.size())));
                nulls[i] = false;
            }
        }
        // Create array with proper null bitmap support
        int dims[1] = {static_cast<int>(num_elements)};
        int lbs[1] = {1}; // Lower bound

        ArrayType* pg_array = construct_md_array(elements,                    // Values
                                                 has_nulls ? nulls : nullptr, // Null bitmap
                                                 1,                           // Number of dimensions
                                                 dims,                        // Dimensions array
                                                 lbs,                         // Lower bounds
                                                 element_typeid,              // Element OID type
                                                 -1,                          // Element size (-1 for variable length)
                                                 false,                       // Pass by value? (false for bytea)
                                                 'i'                          // Element alignment
        );
        return PointerGetDatum(pg_array);
    }
    default: {
        throw pg::exception("Unsupported type for conversion");
        return (Datum)0;
    }
    }
}

// Use concepts!
template <typename T>
inline Datum pointer_to_datum(const void* curr_val, Oid attr_typeid, int32_t attr_typmod, size_t row)
{
    const auto val = reinterpret_cast<const T*>(curr_val)[row];
    switch (attr_typeid) {
    case BOOLOID: {
        return BoolGetDatum(static_cast<bool>(val));
    }
    case INT2OID: {
        return Int16GetDatum(static_cast<int16_t>(val));
    }
    case INT4OID: {
        return Int32GetDatum(static_cast<int32_t>(val));
    }
    case INT8OID: {
        return Int64GetDatum(static_cast<int64_t>(val));
    }
    case FLOAT4OID: {
        return Float4GetDatum(static_cast<float>(val));
    }
    case FLOAT8OID: {
        return Float8GetDatum(static_cast<double>(val));
    }
    case NUMERICOID: {
        if constexpr (std::is_same_v<T, int16_t>) {
            return DirectFunctionCall1(int2_numeric, Int16GetDatum(val));
        } else if constexpr (std::is_same_v<T, int32_t>) {
            return DirectFunctionCall1(int4_numeric, Int32GetDatum(val));
        } else if constexpr (std::is_same_v<T, int64_t>) {
            return DirectFunctionCall1(int8_numeric, Int64GetDatum(val));
        } else if constexpr (std::is_same_v<T, float>) {
            return DirectFunctionCall1(float4_numeric, Float4GetDatum(val));
        } else if constexpr (std::is_same_v<T, double>) {
            return DirectFunctionCall1(float8_numeric, Float8GetDatum(val));
        } else {
            throw pg::exception("Unsupported numeric type for conversion");
        }
    }
    case DATEOID: {
        // Convert from Unix epoch (1970) back to PostgreSQL epoch (2000)
        return DateADTGetDatum(static_cast<DateADT>(val) - pg::POSTGRES_EPOCH_DAYS);
    }
    case TIMEOID: {
        return TimeADTGetDatum(static_cast<TimeADT>(val));
    }
    case TIMESTAMPOID: {
        // Convert from Unix epoch (1970) back to PostgreSQL epoch (2000)
        return TimestampGetDatum(static_cast<Timestamp>(val) - pg::TIMESTAMP_EPOCH_DIFF_US);
    }
    case TIMESTAMPTZOID: {
        // Convert from Unix epoch (1970) back to PostgreSQL epoch (2000)
        return TimestampTzGetDatum(static_cast<TimestampTz>(val) - pg::TIMESTAMP_EPOCH_DIFF_US);
    }
    default: {
        const char* tname = format_type_with_typemod(attr_typeid, attr_typmod);
        throw pg::exception(fmt::format("Non-numeric type '{}' for conversion", tname));
        return (Datum)0;
    }
    }
}

template <typename T>
nd::array eval_with_nones(nd::array arr)
{
    try {
        return nd::eval(arr);
    } catch (const nd::invalid_dynamic_eval&) {
    }
    icm::vector<nd::array> result_elements;
    result_elements.reserve(arr.size());
    for (auto a : arr) {
        if (a.is_none()) {
            result_elements.push_back(nd::adapt(T()));
        } else {
            result_elements.push_back(std::move(a));
        }
    }
    return nd::eval(nd::dynamic(std::move(result_elements)));
}

} // namespace pg::utils
