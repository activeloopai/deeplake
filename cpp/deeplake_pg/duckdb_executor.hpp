#pragma once

#include <deeplake_api/dataset_view.hpp>

#include <memory>
#include <string>

namespace pg {

// Execute SQL query using DuckDB with ZERO-COPY access to table_data
// This version uses custom table function to read directly from table_data streamers
std::shared_ptr<deeplake_api::dataset_view> execute_sql_query(const std::string& query_string);

} // namespace pg
