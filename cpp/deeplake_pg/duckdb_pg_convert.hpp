#pragma once

#include <postgres.h>

#include <cstddef>
#include <cstdint>

namespace duckdb {
class Vector;
class LogicalType;
} // namespace duckdb

namespace pg {

Datum duckdb_value_to_pg_datum(
    const duckdb::Vector& vec, size_t row, Oid target_type, int32_t attr_typmod, bool& is_null);

} // namespace pg
