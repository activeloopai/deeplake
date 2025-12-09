#pragma once

#include <cstddef>
#include <cstdint>

using Oid = unsigned int;
using Datum = unsigned long;

namespace duckdb {
class Vector;
class LogicalType;
} // namespace duckdb

namespace pg {

Datum duckdb_value_to_pg_datum(
    const duckdb::Vector& vec, size_t row, Oid target_type, int32_t attr_typmod, bool& is_null);

} // namespace pg
