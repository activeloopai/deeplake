#pragma once

#include <deeplake_api/dataset_view.hpp>

#include <memory>
#include <string>
#include <utility>
#include <vector>

// Include DuckDB C API header
#include <duckdb.h>

namespace pg {

// Forward declarations - full definitions in .cpp file which includes <duckdb.hpp>
// Using global namespace to avoid conflicts

// Structure to hold DuckDB query results
// Currently uses C++ API internally due to table function requirements
// Structure is ready for C API migration when table functions are supported
struct duckdb_result_holder
{
    // Internal C++ API structures (will be replaced with C API when possible)
    // Using void* to avoid exposing C++ types in header
    void* query_result_ptr = nullptr; // duckdb::QueryResult*
    void* chunks_ptr = nullptr; // std::vector<std::unique_ptr<duckdb::DataChunk>>*
    
    size_t total_rows = 0;
    size_t column_count = 0;

    // Constructor and destructor
    duckdb_result_holder();
    ~duckdb_result_holder();

    // Move semantics
    duckdb_result_holder(duckdb_result_holder&& other) noexcept;
    duckdb_result_holder& operator=(duckdb_result_holder&& other) noexcept;

    // No copy
    duckdb_result_holder(const duckdb_result_holder&) = delete;
    duckdb_result_holder& operator=(const duckdb_result_holder&) = delete;

    // Helper: Get chunk index and row offset for a given global row number
    std::pair<size_t, size_t> get_chunk_and_offset(size_t global_row) const;
    
    // Get number of chunks
    size_t get_chunk_count() const;
    
    // Get column count
    size_t get_column_count() const;
    
    // Get a specific chunk's DataChunk pointer (for C++ API access)
    // Returns nullptr if invalid
    void* get_chunk_ptr(size_t chunk_idx) const;
};

// Execute SQL query using DuckDB and return results directly without conversion
duckdb_result_holder execute_sql_query_direct(const std::string& query_string);

// Legacy function for backward compatibility - converts to dataset_view
// TODO: Remove once all callers are updated
std::shared_ptr<deeplake_api::dataset_view> execute_sql_query(const std::string& query_string);

} // namespace pg
