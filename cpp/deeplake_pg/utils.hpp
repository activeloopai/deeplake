#pragma once

#include <base/getenv.hpp>
#include <fmt/format.h>
#include <nd/array.hpp>

#include <exception>
#include <charconv>

#ifdef __cplusplus
extern "C" {
#endif

/// postgres.h should be included before any other PostgreSQL header
#include <postgres.h>

#include <access/skey.h>
#include <access/tupdesc.h>
#include <catalog/pg_type.h>
#include <executor/spi.h>
#include <funcapi.h>
#include <miscadmin.h>
#include <storage/bufmgr.h>
#include <utils/builtins.h>
#include <utils/guc.h>
#include <utils/lsyscache.h>
#include <utils/memutils.h>
#include <utils/syscache.h>
#include <utils/varlena.h>

#ifdef __cplusplus
} /// extern "C"
#endif

namespace pg {

using array_type = nd::array;

inline static constexpr int32_t POSTGRES_EPOCH_DAYS = 10957;
inline static constexpr int64_t TIMESTAMP_EPOCH_DIFF_US = 946684800000000LL;
inline static constexpr int64_t DEEPLAKE_TUPLES_PER_BLOCK = 256;
inline static constexpr const char* dataset_path_option_name = "dataset_path";
inline static constexpr const char* index_type_option_name = "index_type";

// GUC variables - defined in extension_init.cpp
extern bool use_parallel_workers;
extern bool use_deeplake_executor;
extern bool explain_query_before_execute;
extern bool ignore_primary_keys;
extern bool print_runtime_stats;
extern bool is_filter_pushdown_enabled;
extern int32_t max_streamable_column_width;
extern int32_t max_num_threads_for_global_state;
extern bool treat_numeric_as_double;
extern bool print_progress_during_seq_scan;
extern bool use_shared_mem_for_refresh;
extern bool enable_dataset_logging;

namespace utils {

/**
 * @brief Get the base type OID for a given type, resolving domain types
 * @param typid The type OID to resolve
 * @return The base type OID (same as input if not a domain)
 *
 * For domain types (TYPTYPE_DOMAIN), this returns the underlying base type.
 * For regular types, returns the same OID.
 * This is essential for handling domain types (e.g., IMAGE over BYTEA) correctly.
 */
inline Oid get_base_type(Oid typid)
{
    // Fast path for invalid OID
    if (typid == InvalidOid) {
        return typid;
    }

    // Look up the type in the system cache
    HeapTuple tup = SearchSysCache1(TYPEOID, ObjectIdGetDatum(typid));
    if (!HeapTupleIsValid(tup)) {
        return typid; // Type not found, return as-is
    }

    Form_pg_type typTup = (Form_pg_type) GETSTRUCT(tup);

    // If it's a domain, recursively get the base type
    Oid result = InvalidOid;
    if (typTup->typtype == TYPTYPE_DOMAIN) {
        result = get_base_type(typTup->typbasetype);
    } else {
        result = typid;
    }

    ReleaseSysCache(tup);
    return result;
}

/**
 * @brief Get the element type for array types, resolving domains
 * @param typid The array type OID
 * @return The element type OID, with domains resolved to base types
 */
inline Oid get_base_array_element_type(Oid typid)
{
    Oid elem_type = get_element_type(typid);
    if (OidIsValid(elem_type)) {
        return get_base_type(elem_type);
    }
    return InvalidOid;
}

/// Error handling wrapper
template <typename Func>
inline auto pg_try(Func&& f) -> decltype(f())
{
    try {
        return f();
    } catch (const std::exception& e) {
        ereport(ERROR, (errcode(ERRCODE_INTERNAL_ERROR), errmsg("%s", e.what())));
    }
}

/**
 * @brief RAII wrapper for memory context switch
 * @details This struct is used to switch memory context and reset it
 */
struct memory_context_switcher
{
    memory_context_switcher()
    {
        new_context_ = AllocSetContextCreate(CurrentMemoryContext, "mem_context", ALLOCSET_DEFAULT_SIZES);
        old_context_ = MemoryContextSwitchTo(new_context_);
    }

    memory_context_switcher(MemoryContext ctx)
        : old_context_(MemoryContextSwitchTo(ctx))
    {
    }

    memory_context_switcher(const memory_context_switcher&) = delete;
    memory_context_switcher& operator=(const memory_context_switcher&) = delete;
    memory_context_switcher(memory_context_switcher&&) = delete;
    memory_context_switcher& operator=(memory_context_switcher&&) = delete;

    ~memory_context_switcher()
    {
        MemoryContextSwitchTo(old_context_);
        if (new_context_ != nullptr) {
            MemoryContextDelete(new_context_);
        }
    }

    void reset()
    {
        MemoryContextSwitchTo(old_context_);
        MemoryContextReset(new_context_);
        MemoryContextSwitchTo(new_context_);
    }

private:
    MemoryContext new_context_ = nullptr;
    MemoryContext old_context_ = nullptr;
};

/**
 * @brief RAII wrapper for SPI_connect and SPI_finish
 * @details This class is used to connect to SPI manager and disconnect from it
 * @note Should add support for SPI_execute
 */
struct spi_connector
{
    spi_connector()
    {
        if (SPI_connect() != SPI_OK_CONNECT) {
            elog(ERROR, "Could not connect to SPI manager");
        }
        s_exec_in_progress = true;
    }

    spi_connector(const spi_connector&) = delete;
    spi_connector& operator=(const spi_connector&) = delete;
    spi_connector(spi_connector&&) = delete;
    spi_connector& operator=(spi_connector&&) = delete;

    ~spi_connector()
    {
        SPI_finish();
        s_exec_in_progress = false;
    }

    inline static bool is_execution_in_progress() noexcept
    {
        return s_exec_in_progress;
    }

    inline static bool s_exec_in_progress = false;
};

struct parallel_workers_switcher
{
    parallel_workers_switcher(int32_t num_workers = 0) noexcept
        : num_workers_(max_parallel_maintenance_workers)
    {
        max_parallel_maintenance_workers = num_workers;
    }

    ~parallel_workers_switcher() noexcept
    {
        max_parallel_maintenance_workers = num_workers_;
    }

private:
    int32_t num_workers_ = 0;
};

static std::string get_pg_data_directory()
{
    const char* data_dir = GetConfigOption("data_directory", true, false);
    if (data_dir == nullptr) {
        ereport(ERROR, (errcode(ERRCODE_INTERNAL_ERROR), errmsg("Unable to retrieve data_directory")));
    }
    return std::string(data_dir);
}

static std::string get_deeplake_root_directory()
{
    static const std::string root_dir_variable_name = "DEEPLAKE_ROOT_PATH";
    static const std::string pg_data_dir = get_pg_data_directory();
    static const std::string deeplake_root_dir = base::getenv<std::string>(root_dir_variable_name, pg_data_dir);
    return deeplake_root_dir;
}

inline std::pair<BlockNumber, OffsetNumber> row_number_to_tid(int64_t row_number)
{
    BlockNumber block_number = static_cast<BlockNumber>(row_number / DEEPLAKE_TUPLES_PER_BLOCK);
    OffsetNumber offset = static_cast<OffsetNumber>((row_number % DEEPLAKE_TUPLES_PER_BLOCK) + 1);
    return {block_number, offset};
}

inline int64_t tid_to_row_number(ItemPointer tid)
{
    BlockNumber block_number = ItemPointerGetBlockNumber(tid);
    OffsetNumber offset = ItemPointerGetOffsetNumber(tid);
    return (static_cast<int64_t>(block_number) * DEEPLAKE_TUPLES_PER_BLOCK) + (offset - 1);
}

struct parsed_special_datum_result
{
    bool is_valid = false;
    Oid table_id = InvalidOid;
    AttrNumber column_id = InvalidAttrNumber;
    int64_t row_id = -1;
};

inline bool is_string_type(Oid attr_typeid)
{
    // Check if the type is one of the text-like types
    return (attr_typeid == TEXTOID ||
            attr_typeid == VARCHAROID ||
            attr_typeid == BPCHAROID ||
            attr_typeid == JSONOID ||
            attr_typeid == JSONBOID);
}

inline bool can_make_special_datum(Oid attr_typeid)
{
    bool is_array = type_is_array(attr_typeid);
    return is_array || is_string_type(attr_typeid) || attr_typeid == BYTEAOID;
}

inline uint64_t get_column_width(Oid attr_typeid, int32_t atttypmod)
{
    // Resolve domain types to their base type
    switch (attr_typeid) {
    case INT2OID:
        return sizeof(int16_t);
    case INT4OID:
    case DATEOID:
        return sizeof(int32_t);
    case INT8OID:
    case TIMESTAMPOID:
    case TIMESTAMPTZOID:
    case TIMEOID:
        return sizeof(int64_t);
    case FLOAT4OID:
        return sizeof(float);
    case FLOAT8OID:
        return sizeof(double);
    case BOOLOID:
        return sizeof(bool);
    case CHAROID:
    case VARCHAROID:
    case BPCHAROID:
        return atttypmod;
    case TEXTOID:
        return 256; // Assuming average text length, heuristic
    case UUIDOID:
        return 36;
    case JSONOID:
    case JSONBOID:
        return 64; // Assuming average JSON document size, heuristic
    case BYTEAOID:
        return 512; // Assuming average binary data size, heuristic
    case NUMERICOID:
        return sizeof(double); // We treat NUMERIC as double internally
    // Array types
    case INT2ARRAYOID:
        return 32 * sizeof(int16_t); // Assuming average array length of 32
    case INT4ARRAYOID:
        return 32 * sizeof(int32_t);
    case INT8ARRAYOID:
        return 32 * sizeof(int64_t);
    case FLOAT4ARRAYOID:
        return 32 * sizeof(float);
    case FLOAT8ARRAYOID:
        return 32 * sizeof(double);
    case TEXTARRAYOID:
    case VARCHARARRAYOID:
        return 32 * 256; // 32 text elements of 256 bytes each
    case BYTEAARRAYOID:
        return 32 * 512; // 32 bytea elements of 512 bytes each
    default:
        return 0; // Unsupported type
    }
    return 0;
}

inline constexpr int32_t g_not_fetched_magic = 0xDEADBEAF;

// Helper function to parse space-separated special datum format: "magic table_id row_id column_id"
inline parsed_special_datum_result parse_space_separated_values(const char* data, size_t len)
{
    parsed_special_datum_result result;

    // Parse space-separated values using std::from_chars for performance
    const char* ptr = data;
    const char* end = data + len;

    // Parse magic number
    int32_t magic = 0;
    auto [p1, ec1] = std::from_chars(ptr, end, magic);
    if (ec1 != std::errc{} || magic != g_not_fetched_magic) {
        return result;
    }

    // Skip space
    if (p1 >= end || *p1 != ' ') return result;
    ptr = p1 + 1;

    // Parse table_id
    uint32_t table_id = 0;
    auto [p2, ec2] = std::from_chars(ptr, end, table_id);
    if (ec2 != std::errc{}) return result;

    // Skip space
    if (p2 >= end || *p2 != ' ') return result;
    ptr = p2 + 1;

    // Parse row_id
    int64_t row_id = 0;
    auto [p3, ec3] = std::from_chars(ptr, end, row_id);
    if (ec3 != std::errc{}) return result;

    // Skip space
    if (p3 >= end || *p3 != ' ') return result;
    ptr = p3 + 1;

    // Parse column_id
    int32_t column_id = 0;
    auto [p4, ec4] = std::from_chars(ptr, end, column_id);
    if (ec4 != std::errc{}) return result;

    result.is_valid = true;
    result.table_id = table_id;
    result.row_id = row_id;
    result.column_id = static_cast<AttrNumber>(column_id);

    return result;
}

// Helper function to parse a single bytea element as a number
template<typename T>
inline bool parse_bytea_element(bytea* b, T& value)
{
    if (b == nullptr) return false;

    const char* data = VARDATA(b);
    size_t len = VARSIZE(b) - VARHDRSZ;

    auto [ptr, ec] = std::from_chars(data, data + len, value);
    return ec == std::errc{};
}

inline Datum make_special_datum(Oid table_id, int64_t row_id, AttrNumber column_id, Oid attr_typeid)
{
    if (!can_make_special_datum(attr_typeid)) {
        elog(ERROR, "Cannot create special datum for type %u", attr_typeid);
    }
    if (is_string_type(attr_typeid)) {
        std::string str = fmt::format("{} {} {} {}", g_not_fetched_magic, table_id, row_id, column_id);
        return PointerGetDatum(cstring_to_text_with_len(str.data(), static_cast<int>(str.size())));
    } else if (type_is_array(attr_typeid)) {
        switch (attr_typeid)
        {
        case INT2ARRAYOID: {
            Datum* elements = (Datum*)palloc(8 * sizeof(Datum));
            elements[0] = Int16GetDatum(static_cast<int16_t>(g_not_fetched_magic & 0xFFFF));
            elements[1] = Int16GetDatum(static_cast<int16_t>((g_not_fetched_magic >> 16) & 0xFFFF));
            elements[2] = Int16GetDatum(static_cast<int16_t>(table_id & 0xFFFF));
            elements[3] = Int16GetDatum(static_cast<int16_t>((table_id >> 16) & 0xFFFF));
            elements[4] = Int16GetDatum(static_cast<int16_t>(row_id & 0xFFFF));
            elements[5] = Int16GetDatum(static_cast<int16_t>((row_id >> 16) & 0xFFFF));
            elements[6] = Int16GetDatum(static_cast<int16_t>((row_id >> 32) & 0xFFFF));
            elements[7] = Int16GetDatum(static_cast<int16_t>(column_id));
            return PointerGetDatum(construct_array(elements, 8, INT2OID, sizeof(int16_t), true, 'i'));
        }
        case INT4ARRAYOID: {
            Datum* elements = (Datum*)palloc(4 * sizeof(Datum));
            elements[0] = Int32GetDatum(g_not_fetched_magic);
            elements[1] = Int32GetDatum(static_cast<int32_t>(table_id));
            elements[2] = Int64GetDatum(row_id);
            elements[3] = Int32GetDatum(static_cast<int32_t>(column_id));
            return PointerGetDatum(construct_array(elements, 4, INT4OID, sizeof(int32_t), true, 'i'));
        }
        case INT8ARRAYOID: {
            Datum* elements = (Datum*)palloc(4 * sizeof(Datum));
            elements[0] = Int64GetDatum(static_cast<int64_t>(g_not_fetched_magic));
            elements[1] = Int64GetDatum(static_cast<int64_t>(table_id));
            elements[2] = Int64GetDatum(static_cast<int64_t>(row_id));
            elements[3] = Int64GetDatum(static_cast<int64_t>(column_id));
            return PointerGetDatum(construct_array(elements, 4, INT8OID, sizeof(int64_t), true, 'd'));
        }
        case FLOAT4ARRAYOID: {
            Datum* elements = (Datum*)palloc(4 * sizeof(Datum));
            elements[0] = Float4GetDatum(static_cast<float>(g_not_fetched_magic));
            elements[1] = Float4GetDatum(static_cast<float>(table_id));
            elements[2] = Float4GetDatum(static_cast<float>(row_id));
            elements[3] = Float4GetDatum(static_cast<float>(column_id));
            return PointerGetDatum(construct_array(elements, 4, FLOAT4OID, sizeof(float), true, 'f'));
        }
        case FLOAT8ARRAYOID: {
            Datum* elements = (Datum*)palloc(4 * sizeof(Datum));
            elements[0] = Float8GetDatum(static_cast<double>(g_not_fetched_magic));
            elements[1] = Float8GetDatum(static_cast<double>(table_id));
            elements[2] = Float8GetDatum(static_cast<double>(row_id));
            elements[3] = Float8GetDatum(static_cast<double>(column_id));
            return PointerGetDatum(construct_array(elements, 4, FLOAT8OID, sizeof(double), true, 'd'));
        }
        case BYTEAARRAYOID: {
            // For BYTEA arrays, we create an array of bytea elements
            Datum* elements = (Datum*)palloc(4 * sizeof(Datum));

            // Create bytea for magic number
            std::string magic_str = std::to_string(g_not_fetched_magic);
            bytea* magic_b = (bytea*)palloc(static_cast<size_t>(VARHDRSZ) + magic_str.size());
            SET_VARSIZE(magic_b, static_cast<int32_t>(static_cast<size_t>(VARHDRSZ) + magic_str.size()));
            memcpy(VARDATA(magic_b), magic_str.data(), magic_str.size());
            elements[0] = PointerGetDatum(magic_b);

            // Create bytea for table_id
            std::string table_str = std::to_string(table_id);
            bytea* table_b = (bytea*)palloc(static_cast<size_t>(VARHDRSZ) + table_str.size());
            SET_VARSIZE(table_b, static_cast<int32_t>(static_cast<size_t>(VARHDRSZ) + table_str.size()));
            memcpy(VARDATA(table_b), table_str.data(), table_str.size());
            elements[1] = PointerGetDatum(table_b);

            // Create bytea for row_id
            std::string row_str = std::to_string(row_id);
            bytea* row_b = (bytea*)palloc(static_cast<size_t>(VARHDRSZ) + row_str.size());
            SET_VARSIZE(row_b, static_cast<int32_t>(static_cast<size_t>(VARHDRSZ) + row_str.size()));
            memcpy(VARDATA(row_b), row_str.data(), row_str.size());
            elements[2] = PointerGetDatum(row_b);

            // Create bytea for column_id
            std::string col_str = std::to_string(column_id);
            bytea* col_b = (bytea*)palloc(static_cast<size_t>(VARHDRSZ) + col_str.size());
            SET_VARSIZE(col_b, static_cast<int32_t>(static_cast<size_t>(VARHDRSZ) + col_str.size()));
            memcpy(VARDATA(col_b), col_str.data(), col_str.size());
            elements[3] = PointerGetDatum(col_b);

            return PointerGetDatum(construct_array(elements, 4, BYTEAOID, -1, false, 'i'));
        }
        default:
            elog(ERROR, "Unsupported array type for special datum: %u", attr_typeid);
            break;
        }
    } else if (attr_typeid == BYTEAOID) {
        // For BYTEA, we can use a bytea type
        std::string str = fmt::format("{} {} {} {}", g_not_fetched_magic, table_id, row_id, column_id);
        bytea* b = (bytea*)palloc(static_cast<size_t>(VARHDRSZ) + str.size());
        SET_VARSIZE(b, static_cast<int32_t>(static_cast<size_t>(VARHDRSZ) + str.size()));
        memcpy(VARDATA(b), str.data(), str.size());
        return PointerGetDatum(b);
    }
    return (Datum)0; // Should never reach here
}

inline parsed_special_datum_result parse_special_datum(Datum d, Oid attr_typeid)
{
    parsed_special_datum_result result;

    if (!can_make_special_datum(attr_typeid)) {
        return result;
    }

    if (is_string_type(attr_typeid)) {
        // For string types, parse the formatted string: "magic table_id row_id column_id"
        text* txt = DatumGetTextP(d);
        if (txt == nullptr) {
            return result;
        }

        const char* data = VARDATA(txt);
        size_t len = VARSIZE(txt) - VARHDRSZ;

        result = parse_space_separated_values(data, len);

    } else if (type_is_array(attr_typeid)) {
        // For array types, check the first element for magic and parse accordingly
        ArrayType* arr = DatumGetArrayTypeP(d);
        if (arr == nullptr) {
            return result;
        }

        int ndims = ARR_NDIM(arr);
        if (ndims != 1) {
            return result;
        }

        int* dims = ARR_DIMS(arr);
        if (dims[0] < 4) {
            return result;
        }

        switch (attr_typeid) {
        case INT2ARRAYOID: {
            if (dims[0] != 8) return result;

            int16* data = (int16*)ARR_DATA_PTR(arr);

            // Reconstruct magic number from first two elements
            int32_t magic = static_cast<int32_t>(data[0]) | (static_cast<int32_t>(data[1]) << 16);
            if (magic != g_not_fetched_magic) {
                return result;
            }

            // Reconstruct table_id and row_id from split parts
            uint32_t table_id = static_cast<uint32_t>(data[2]) | (static_cast<uint32_t>(data[3]) << 16);
            int64_t row_id = static_cast<int64_t>(data[4]) |
                           (static_cast<int64_t>(data[5]) << 16) |
                           (static_cast<int64_t>(data[6]) << 32);

            result.is_valid = true;
            result.table_id = table_id;
            result.row_id = row_id;
            result.column_id = static_cast<AttrNumber>(data[7]);
            break;
        }
        case INT4ARRAYOID: {
            int32* data = (int32*)ARR_DATA_PTR(arr);
            if (data[0] != g_not_fetched_magic) {
                return result;
            }

            result.is_valid = true;
            result.table_id = static_cast<uint32_t>(data[1]);
            result.row_id = static_cast<int64_t>(data[2]);
            result.column_id = static_cast<AttrNumber>(data[3]);
            break;
        }
        case INT8ARRAYOID: {
            int64* data = (int64*)ARR_DATA_PTR(arr);
            if (data[0] != static_cast<int64_t>(g_not_fetched_magic)) {
                return result;
            }

            result.is_valid = true;
            result.table_id = static_cast<uint32_t>(data[1]);
            result.row_id = data[2];
            result.column_id = static_cast<AttrNumber>(data[3]);
            break;
        }
        case FLOAT4ARRAYOID: {
            float* data = (float*)ARR_DATA_PTR(arr);
            if (static_cast<int32_t>(data[0]) != g_not_fetched_magic) {
                return result;
            }

            result.is_valid = true;
            result.table_id = static_cast<uint32_t>(data[1]);
            result.row_id = static_cast<int64_t>(data[2]);
            result.column_id = static_cast<AttrNumber>(data[3]);
            break;
        }
        case FLOAT8ARRAYOID: {
            double* data = (double*)ARR_DATA_PTR(arr);
            if (static_cast<int32_t>(data[0]) != g_not_fetched_magic) {
                return result;
            }

            result.is_valid = true;
            result.table_id = static_cast<uint32_t>(data[1]);
            result.row_id = static_cast<int64_t>(data[2]);
            result.column_id = static_cast<AttrNumber>(data[3]);
            break;
        }
        case BYTEAARRAYOID: {
            // For bytea arrays, parse each element as string
            Datum* elem_datums = nullptr;
            bool* elem_nulls = nullptr;
            int elem_count = 0;
            deconstruct_array(arr, BYTEAOID, -1, false, 'i', &elem_datums, &elem_nulls, &elem_count);

            if (elem_count != 4 || elem_nulls[0] || elem_nulls[1] || elem_nulls[2] || elem_nulls[3]) {
                return result;
            }

            // Parse magic from first element
            int32_t magic = 0;
            if (!parse_bytea_element(DatumGetByteaP(elem_datums[0]), magic) || magic != g_not_fetched_magic) {
                return result;
            }

            // Parse table_id, row_id, column_id from remaining elements
            uint32_t table_id = 0;
            int64_t row_id = 0;
            int32_t column_id = 0;

            if (!parse_bytea_element(DatumGetByteaP(elem_datums[1]), table_id) ||
                !parse_bytea_element(DatumGetByteaP(elem_datums[2]), row_id) ||
                !parse_bytea_element(DatumGetByteaP(elem_datums[3]), column_id)) {
                return result;
            }

            result.is_valid = true;
            result.table_id = table_id;
            result.row_id = row_id;
            result.column_id = static_cast<AttrNumber>(column_id);
            break;
        }
        }
    } else if (attr_typeid == BYTEAOID) {
        // For single bytea, parse the formatted string
        bytea* b = DatumGetByteaP(d);
        if (b == nullptr) {
            return result;
        }

        const char* data = VARDATA(b);
        size_t len = VARSIZE(b) - VARHDRSZ;

        result = parse_space_separated_values(data, len);
    }

    return result;
}

inline bool check_table_exists(const std::string& table_name, const std::string& schema_name = "public")
{
    std::string query = fmt::format("SELECT 1 FROM information_schema.tables WHERE "
                                    "table_schema = '{}' AND table_name = '{}'",
                                    schema_name, table_name);
    spi_connector connector;
    return SPI_execute(query.c_str(), true, 0) == SPI_OK_SELECT && SPI_processed > 0;
}

inline bool check_column_exists(const std::string& table_name, const std::string& column_name, const std::string& schema_name = "public")
{
    std::string query = fmt::format("SELECT 1 FROM information_schema.columns WHERE "
                                    "table_schema = '{}' AND table_name = '{}' AND column_name = '{}'",
                                    schema_name, table_name, column_name);
    spi_connector connector;
    return SPI_execute(query.c_str(), true, 0) == SPI_OK_SELECT && SPI_processed > 0;
}

} // namespace utils

} // namespace pg
