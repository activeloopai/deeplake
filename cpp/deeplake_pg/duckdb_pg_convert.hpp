#pragma once

#include <cstddef>
#include <cstdint>

using Oid = unsigned int;
using Datum = unsigned long;

// Forward declaration - we use C++ API Vector internally
// Note: Full definition is in duckdb_pg_convert.cpp which includes <duckdb.hpp>
namespace duckdb {
class Vector;
} // namespace duckdb

namespace pg {

// Convert a value from DuckDB Vector to PostgreSQL Datum
// Note: Currently uses C++ API Vector (required for table functions)
// Will be updated to use C API duckdb_result when table functions are supported via C API
Datum duckdb_value_to_pg_datum(
    const ::duckdb::Vector& vec, size_t row, Oid target_type, int32_t attr_typmod, bool& is_null);

} // namespace pg
