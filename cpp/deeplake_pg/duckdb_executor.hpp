#pragma once

#include <deeplake_api/dataset_view.hpp>

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace duckdb {
class QueryResult;
class DataChunk;
} // namespace duckdb

namespace pg {

// Structure to hold DuckDB query results without intermediate conversion
// Destructor must be defined in .cpp where DuckDB types are complete
struct duckdb_result_holder
{
    std::unique_ptr<duckdb::QueryResult> query_result;
    std::vector<std::unique_ptr<duckdb::DataChunk>> chunks;
    size_t total_rows = 0;

    // Constructor and destructor must be in .cpp file for PIMPL pattern
    duckdb_result_holder();
    ~duckdb_result_holder();

    // Move semantics
    duckdb_result_holder(duckdb_result_holder&&) noexcept;
    duckdb_result_holder& operator=(duckdb_result_holder&&) noexcept;

    // No copy
    duckdb_result_holder(const duckdb_result_holder&) = delete;
    duckdb_result_holder& operator=(const duckdb_result_holder&) = delete;

    // Helper: Get chunk index and row offset for a given global row number
    std::pair<size_t, size_t> get_chunk_and_offset(size_t global_row) const;
};

// Execute SQL query using DuckDB and return results directly without conversion
duckdb_result_holder execute_sql_query_direct(const std::string& query_string);

// Legacy function for backward compatibility - converts to dataset_view
// TODO: Remove once all callers are updated
std::shared_ptr<deeplake_api::dataset_view> execute_sql_query(const std::string& query_string);

} // namespace pg
