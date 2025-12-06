#pragma once

// Forward declarations
namespace duckdb {
class Connection;
}

namespace pg {

// Register the deeplake_scan table function with DuckDB
// This allows querying table_data without copying:
// SELECT * FROM deeplake_scan(ptr_to_table_data, 'table_name')
void register_deeplake_scan_function(duckdb::Connection& con);

} // namespace pg
