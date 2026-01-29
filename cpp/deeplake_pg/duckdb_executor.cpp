// Include libintl.h first to avoid conflicts with PostgreSQL's gettext macro
#include <libintl.h>

// DuckDB headers must come before PostgreSQL headers to avoid namespace pollution
// Use C++ API only for table function registration (required)
#include <duckdb.hpp>
// Use C API for query execution (public API)
#include <duckdb.h>

#include "duckdb_deeplake_convert.hpp"
#include "duckdb_deeplake_scan.hpp"
#include "duckdb_executor.hpp"
#include "pg_deeplake.hpp"
#include "pg_to_duckdb_translator.hpp"
#include "reporter.hpp"
#include "table_data.hpp"
#include "table_storage.hpp"
#include "utils.hpp"

#include <memory>
#include <string>
#include <vector>
#include <cstring>

namespace {

// Structure to hold both C++ and C API connections
// C++ connection is needed for table function registration
// C connection is used for query execution (public API)
struct duckdb_connections {
    std::unique_ptr<duckdb::DuckDB> db_cpp;
    std::unique_ptr<duckdb::Connection> con_cpp;
    duckdb_database db_c;
    duckdb_connection con_c;

    duckdb_connections() : db_c(nullptr), con_c(nullptr) {}

    ~duckdb_connections() {
        if (con_c) {
            duckdb_disconnect(&con_c);
        }
        if (db_c) {
            duckdb_close(&db_c);
        }
    }

    // No copy
    duckdb_connections(const duckdb_connections&) = delete;
    duckdb_connections& operator=(const duckdb_connections&) = delete;

    // Move semantics
    duckdb_connections(duckdb_connections&& other) noexcept
        : db_cpp(std::move(other.db_cpp))
        , con_cpp(std::move(other.con_cpp))
        , db_c(other.db_c)
        , con_c(other.con_c)
    {
        other.db_c = nullptr;
        other.con_c = nullptr;
    }
};

std::unique_ptr<duckdb_connections> create_connections()
{
    try {
        auto conns = std::make_unique<duckdb_connections>();

        // Create C++ database and connection for table function registration
        duckdb::DBConfig config;
        config.options.allow_unsigned_extensions = true;
        conns->db_cpp = std::make_unique<duckdb::DuckDB>(":memory:", &config);
        conns->con_cpp = std::make_unique<duckdb::Connection>(*(conns->db_cpp));

        // Register the deeplake_scan table function for zero-copy access
        pg::register_deeplake_scan_function(*(conns->con_cpp));

        // For now, we'll use C++ API for queries since table functions require it
        // The C API connection will be used later when we can restructure to avoid table functions
        // or when DuckDB provides a way to register table functions via C API
        // For now, set to nullptr to indicate C API is not yet used
        conns->db_c = nullptr;
        conns->con_c = nullptr;

        return conns;
    } catch (const std::exception& e) {
        elog(ERROR, "Failed to create DuckDB connections: %s", e.what());
    } catch (...) {
        elog(ERROR, "Failed to create DuckDB connections: unknown error");
    }
    return nullptr;
}

void register_table(duckdb_connections* conns, const std::string& table_name, Oid table_id)
{
    try {
        size_t dot_pos = table_name.find('.');
        ASSERT(dot_pos != std::string::npos);
        auto schema_name = table_name.substr(0, dot_pos);

        // Create a VIEW that calls deeplake_scan with pointer to table_data
        std::string create_view_sql =
            fmt::format("CREATE OR REPLACE VIEW \"{}\".\"{}\" AS SELECT * FROM deeplake_scan(CAST({} AS UINTEGER))",
                        schema_name,
                        table_name.substr(dot_pos + 1),
                        table_id);

        // Register in C++ connection (for table function)
        conns->con_cpp->BeginTransaction();
        auto schema_query = fmt::format("CREATE SCHEMA IF NOT EXISTS \"{}\"", schema_name);
        auto schema_result = conns->con_cpp->Query(schema_query);
        if (schema_result->HasError()) {
            conns->con_cpp->Rollback();
            elog(ERROR, "Failed to create schema: %s", schema_result->GetError().c_str());
            return;
        }
        auto result = conns->con_cpp->Query(create_view_sql);
        if (result->HasError()) {
            conns->con_cpp->Rollback();
            elog(ERROR, "Failed to create view: %s", result->GetError().c_str());
            return;
        }
        conns->con_cpp->Commit();

        // Also register in C connection (for queries)
        // Note: C connection can't use deeplake_scan table function directly,
        // so we'll need to execute queries via C++ connection for now
        // This is a limitation we'll work around
    } catch (const std::exception& e) {
        elog(ERROR, "Failed to register table (zero-copy): %s", e.what());
    } catch (...) {
        elog(ERROR, "Failed to register table (zero-copy)");
    }
}

void register_views(duckdb_connections* conns)
{
    try {
        conns->con_cpp->BeginTransaction();
        const auto set_search_path =
            fmt::format("SET search_path TO \"{}\"", pg::table_storage::instance().get_schema_name());
        if (auto res = conns->con_cpp->Query(set_search_path); res->HasError()) {
            conns->con_cpp->Rollback();
            conns->con_cpp->BeginTransaction();
        }
        if (auto res = conns->con_cpp->Query("CREATE TYPE IMAGE AS BLOB"); res->HasError()) {
            conns->con_cpp->Rollback();
            elog(ERROR, "Failed to create type image: %s", res->GetError().c_str());
            return;
        }
        if (auto res = conns->con_cpp->Query("SET autoinstall_known_extensions=1"); res->HasError()) {
            conns->con_cpp->Rollback();
            elog(ERROR, "Failed to set autoinstall_known_extensions: %s", res->GetError().c_str());
            return;
        }
        if (auto res = conns->con_cpp->Query("SET autoload_known_extensions=1"); res->HasError()) {
            conns->con_cpp->Rollback();
            elog(ERROR, "Failed to set autoload_known_extensions: %s", res->GetError().c_str());
            return;
        }
        for (const auto& [_, view_name2query] : pg::table_storage::instance().get_views()) {
            auto result = conns->con_cpp->Query(view_name2query.second);
            if (result->HasError()) {
                conns->con_cpp->Rollback();
                elog(ERROR, "Failed to create view: %s", result->GetError().c_str());
            }
        }
        conns->con_cpp->Commit();
    } catch (const std::exception& e) {
        elog(ERROR, "Failed to register view: %s", e.what());
    }
}

void explain_query(duckdb_connections* conns, const std::string& query_string)
{
    // Use C++ API for explain queries (simpler for now)
    // Check default memory limit
    auto result = conns->con_cpp->Query("SELECT current_setting('memory_limit')");
    if (result && !result->HasError()) {
        elog(INFO, "DuckDB default memory_limit: %s", result->GetValue(0, 0).ToString().c_str());
    }

    // Check threads
    result = conns->con_cpp->Query("SELECT current_setting('threads')");
    if (result && !result->HasError()) {
        elog(INFO, "DuckDB threads: %s", result->GetValue(0, 0).ToString().c_str());
    }

    // Check temp directory (where spilling happens)
    result = conns->con_cpp->Query("SELECT current_setting('temp_directory')");
    if (result && !result->HasError()) {
        elog(INFO, "DuckDB temp_directory: %s", result->GetValue(0, 0).ToString().c_str());
    }

    // Verify critical settings were applied
    result = conns->con_cpp->Query("SELECT current_setting('max_memory')");
    if (result && !result->HasError()) {
        elog(INFO, "DuckDB max_memory: %s", result->GetValue(0, 0).ToString().c_str());
    }

    result = conns->con_cpp->Query("SELECT current_setting('enable_external_access')");
    if (result && !result->HasError()) {
        elog(INFO, "DuckDB enable_external_access: %s", result->GetValue(0, 0).ToString().c_str());
    }

    result = conns->con_cpp->Query("SELECT current_setting('external_threads')");
    if (result && !result->HasError()) {
        elog(INFO, "DuckDB external_threads: %s", result->GetValue(0, 0).ToString().c_str());
    }

    result = conns->con_cpp->Query("SELECT current_setting('temp_block_size')");
    if (result && !result->HasError()) {
        elog(INFO, "DuckDB temp_block_size: %s", result->GetValue(0, 0).ToString().c_str());
    }

    conns->con_cpp->Query("SET enable_profiling = true");
    conns->con_cpp->Query("SET profiling_mode = 'detailed'");

    std::string explain_query_str = "EXPLAIN " + query_string;
    auto explain_result = conns->con_cpp->Query(explain_query_str);

    if (!explain_result->HasError()) {
        elog(INFO, "=== DuckDB Query Plan ===");

        // Get column names to show which plan we're displaying
        auto column_count = explain_result->ColumnCount();

        // Fetch the first chunk (EXPLAIN returns one row with multiple columns)
        auto chunk = explain_result->Fetch();
        if (chunk && chunk->size() > 0) {
            // Iterate through columns (logical_plan, logical_opt, physical_plan, etc.)
            for (size_t col = 0; col < column_count; col++) {
                std::string col_name = explain_result->names[col];
                auto value = chunk->GetValue(col, 0);
                std::string plan_text = value.ToString();

                // Print section header
                elog(INFO, "--- %s ---", col_name.c_str());

                // Print each line of the plan
                size_t start = 0;
                size_t end = 0;
                while ((end = plan_text.find('\n', start)) != std::string::npos) {
                    std::string line = plan_text.substr(start, end - start);
                    elog(INFO, "%s", line.c_str());
                    start = end + 1;
                }
                // Print last line if any
                if (start < plan_text.length()) {
                    std::string line = plan_text.substr(start);
                    elog(INFO, "%s", line.c_str());
                }
            }
        }
        elog(INFO, "=========================");
    }
}

} // unnamed namespace

namespace pg {

// Implementation of duckdb_result_holder methods
// Currently uses C++ API internally due to table function requirements
duckdb_result_holder::duckdb_result_holder() 
    : query_result_ptr(nullptr)
    , chunks_ptr(nullptr)
    , total_rows(0)
    , column_count(0)
{
}

duckdb_result_holder::~duckdb_result_holder()
{
    // Clean up C++ API structures
    // Note: Types are complete here because this file includes <duckdb.hpp>
    if (chunks_ptr) {
        auto* chunks = static_cast<std::vector<std::unique_ptr<duckdb::DataChunk>>*>(chunks_ptr);
        delete chunks;
        chunks_ptr = nullptr;
    }
    if (query_result_ptr) {
        // Use duckdb::unique_ptr which is what DuckDB actually uses
        auto* result = static_cast<duckdb::unique_ptr<duckdb::QueryResult>*>(query_result_ptr);
        delete result;
        query_result_ptr = nullptr;
    }
}

duckdb_result_holder::duckdb_result_holder(duckdb_result_holder&& other) noexcept
    : query_result_ptr(other.query_result_ptr)
    , chunks_ptr(other.chunks_ptr)
    , total_rows(other.total_rows)
    , column_count(other.column_count)
{
    other.query_result_ptr = nullptr;
    other.chunks_ptr = nullptr;
    other.total_rows = 0;
    other.column_count = 0;
}

duckdb_result_holder& duckdb_result_holder::operator=(duckdb_result_holder&& other) noexcept
{
    if (this != &other) {
        // Clean up existing data
        // Note: Types are complete here because this file includes <duckdb.hpp>
        if (chunks_ptr) {
            auto* chunks = static_cast<std::vector<std::unique_ptr<duckdb::DataChunk>>*>(chunks_ptr);
            delete chunks;
        }
        if (query_result_ptr) {
            auto* result = static_cast<duckdb::unique_ptr<duckdb::QueryResult>*>(query_result_ptr);
            delete result;
        }
        
        query_result_ptr = other.query_result_ptr;
        chunks_ptr = other.chunks_ptr;
        total_rows = other.total_rows;
        column_count = other.column_count;
        
        other.query_result_ptr = nullptr;
        other.chunks_ptr = nullptr;
        other.total_rows = 0;
        other.column_count = 0;
    }
    return *this;
}

std::pair<size_t, size_t> duckdb_result_holder::get_chunk_and_offset(size_t global_row) const
{
    if (!chunks_ptr) {
        return {0, global_row};
    }
    auto* chunks = static_cast<std::vector<std::unique_ptr<duckdb::DataChunk>>*>(chunks_ptr);
    size_t accumulated_rows = 0;
    for (size_t chunk_idx = 0; chunk_idx < chunks->size(); ++chunk_idx) {
        size_t chunk_size = (*chunks)[chunk_idx]->size();
        if (global_row < accumulated_rows + chunk_size) {
            return {chunk_idx, global_row - accumulated_rows};
        }
        accumulated_rows += chunk_size;
    }
    return {chunks->size() - 1, chunks->back()->size() - 1};
}

size_t duckdb_result_holder::get_chunk_count() const
{
    if (!chunks_ptr) {
        return 0;
    }
    auto* chunks = static_cast<std::vector<std::unique_ptr<duckdb::DataChunk>>*>(chunks_ptr);
    return chunks->size();
}

size_t duckdb_result_holder::get_column_count() const
{
    return column_count;
}

void* duckdb_result_holder::get_chunk_ptr(size_t chunk_idx) const
{
    if (!chunks_ptr) {
        return nullptr;
    }
    auto* chunks = static_cast<std::vector<std::unique_ptr<duckdb::DataChunk>>*>(chunks_ptr);
    if (chunk_idx >= chunks->size()) {
        return nullptr;
    }
    return (*chunks)[chunk_idx].get();
}

// C API helper methods removed - we're using C++ API internally
// These would be used when migrating to full C API (when table functions are supported)

// Execute SQL query and return DuckDB results using C API
// Note: Currently we still use C++ API for execution due to table function requirements
// This function converts C++ results to C API format for processing
duckdb_result_holder execute_sql_query_direct(const std::string& query_string)
{
    static std::unique_ptr<duckdb_connections> conns;
    if (conns == nullptr || !pg::table_storage::instance().is_up_to_date()) {
        conns = create_connections();
        auto& deeplake_tables = pg::table_storage::instance().get_tables();
        for (const auto& [table_id, table_data] : deeplake_tables) {
            register_table(conns.get(), table_data.get_table_name(), table_id);
        }
        register_views(conns.get());
        pg::table_storage::instance().set_up_to_date(true);
    }

    if (pg::explain_query_before_execute) {
        explain_query(conns.get(), query_string);
    }

    // IMPORTANT LIMITATION: Table functions (deeplake_scan) require C++ API
    // DuckDB C API does not support registering custom table functions
    // Therefore, we must use C++ API for:
    //   1. Table function registration (unavoidable)
    //   2. Query execution (queries use table functions)
    //
    // We minimize C++ API usage to only what's required and structure code
    // to be ready for C API when table functions are supported via C API
    
    elog(DEBUG1, "Executing DuckDB query: %s", query_string.c_str());
    pg::runtime_printer printer("DuckDB query execution");

    duckdb_result_holder holder;

    // Execute query via C++ API (required because queries use table functions)
    auto result_cpp = conns->con_cpp->SendQuery(query_string);
    if (!result_cpp) {
        ereport(ERROR, (errcode(ERRCODE_INTERNAL_ERROR), errmsg("Query execution returned null result")));
    }

    if (result_cpp->HasError()) {
        std::string error_msg = result_cpp->GetError();
        ereport(ERROR, (errcode(ERRCODE_INTERNAL_ERROR), errmsg("Query failed: %s", error_msg.c_str())));
    }

    holder.column_count = result_cpp->ColumnCount();
    holder.total_rows = 0;

    // Store C++ QueryResult in holder
    // SendQuery returns duckdb::unique_ptr<duckdb::QueryResult>
    // We store a pointer to the unique_ptr to avoid exposing C++ types in header
    auto* query_result_storage = new duckdb::unique_ptr<duckdb::QueryResult>();
    *query_result_storage = std::move(result_cpp);
    holder.query_result_ptr = query_result_storage;

    // Fetch all chunks
    auto* chunks_storage = new std::vector<std::unique_ptr<duckdb::DataChunk>>();
    while (true) {
        auto chunk = (*query_result_storage)->Fetch();
        if (!chunk || chunk->size() == 0) {
            break;
        }
        holder.total_rows += chunk->size();
        chunks_storage->push_back(std::move(chunk));
    }
    holder.chunks_ptr = chunks_storage;
    
    elog(DEBUG1, "DuckDB query returned %zu rows with %zu columns in %zu chunks (C++ API required for table functions)", 
         holder.total_rows, holder.column_count, holder.get_chunk_count());
    
    return holder;
}

} // namespace pg
