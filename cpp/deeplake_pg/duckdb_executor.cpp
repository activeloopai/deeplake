#include <duckdb.hpp>

#include "duckdb_executor.hpp"
#include "duckdb_deeplake_convert.hpp"
#include "duckdb_deeplake_scan.hpp"
#include "pg_deeplake.hpp"
#include "reporter.hpp"
#include "table_data.hpp"
#include "table_storage.hpp"
#include "utils.hpp"

#include <memory>
#include <string>
#include <vector>

namespace {

std::unique_ptr<duckdb::Connection> create_connection()
{
    try {
        duckdb::DBConfig config;
        config.options.allow_unsigned_extensions = true;
        auto db = std::make_unique<duckdb::DuckDB>(":memory:", &config);
        auto con = std::make_unique<duckdb::Connection>(*db);

        // Register the deeplake_scan table function for zero-copy access
        pg::register_deeplake_scan_function(*con);

        return con;
    } catch (const std::exception& e) {
        elog(ERROR, "Failed to create DuckDB connection: %s", e.what());
    } catch (...) {
        elog(ERROR, "Failed to create DuckDB connection: unknown error");
    }
    return nullptr;
}

void register_table(duckdb::Connection* con, const std::string& table_name, Oid table_id)
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

        // Start transaction, create view, and commit
        con->BeginTransaction();
        auto schema_query = fmt::format("CREATE SCHEMA IF NOT EXISTS \"{}\"", schema_name);
        auto schema_result = con->Query(schema_query);
        if (schema_result->HasError()) {
            con->Rollback();
            elog(ERROR, "Failed to create schema: %s", schema_result->GetError().c_str());
            return;
        }
        auto result = con->Query(create_view_sql);
        if (result->HasError()) {
            con->Rollback();
            elog(ERROR, "Failed to create view: %s", result->GetError().c_str());
            return;
        }
        con->Commit();
    } catch (const std::exception& e) {
        elog(ERROR, "Failed to register table (zero-copy): %s", e.what());
    } catch (...) {
        elog(ERROR, "Failed to register table (zero-copy)");
    }
}

void register_views(duckdb::Connection* con)
{
    try {
        con->BeginTransaction();
        const auto set_search_path =
            fmt::format("SET search_path TO \"{}\"", pg::table_storage::instance().get_schema_name());
        if (auto res = con->Query(set_search_path); res->HasError()) {
            con->Rollback();
            con->BeginTransaction();
        }
        if (auto res = con->Query("CREATE TYPE IMAGE AS BLOB"); res->HasError()) {
            con->Rollback();
            elog(ERROR, "Failed to create type image: %s", res->GetError().c_str());
            return;
        }
        if (auto res = con->Query("SET autoinstall_known_extensions=1"); res->HasError()) {
            con->Rollback();
            elog(ERROR, "Failed to set autoinstall_known_extensions: %s", res->GetError().c_str());
            return;
        }
        if (auto res = con->Query("SET autoload_known_extensions=1"); res->HasError()) {
            con->Rollback();
            elog(ERROR, "Failed to set autoload_known_extensions: %s", res->GetError().c_str());
            return;
        }
        for (const auto& [_, view_name2query] : pg::table_storage::instance().get_views()) {
            auto result = con->Query(view_name2query.second);
            if (result->HasError()) {
                con->Rollback();
                elog(ERROR, "Failed to create view: %s", result->GetError().c_str());
            }
        }
        con->Commit();
    } catch (const std::exception& e) {
        elog(ERROR, "Failed to register view: %s", e.what());
    }
}

std::unique_ptr<duckdb::QueryResult> execute_query(duckdb::Connection* con, const std::string& query)
{
    auto result = con->SendQuery(query);
    if (result->HasError()) {
        elog(ERROR, "Query failed: %s", result->GetError().c_str());
    }
    return result;
}

void explain_query(duckdb::Connection* con, const std::string& query_string)
{
    // Check default memory limit
    auto result = con->Query("SELECT current_setting('memory_limit')");
    if (result && !result->HasError()) {
        elog(INFO, "DuckDB default memory_limit: %s", result->GetValue(0, 0).ToString().c_str());
    }

    // Check threads
    result = con->Query("SELECT current_setting('threads')");
    if (result && !result->HasError()) {
        elog(INFO, "DuckDB threads: %s", result->GetValue(0, 0).ToString().c_str());
    }

    // Check temp directory (where spilling happens)
    result = con->Query("SELECT current_setting('temp_directory')");
    if (result && !result->HasError()) {
        elog(INFO, "DuckDB temp_directory: %s", result->GetValue(0, 0).ToString().c_str());
    }

    // Verify critical settings were applied
    result = con->Query("SELECT current_setting('max_memory')");
    if (result && !result->HasError()) {
        elog(INFO, "DuckDB max_memory: %s", result->GetValue(0, 0).ToString().c_str());
    }

    result = con->Query("SELECT current_setting('enable_external_access')");
    if (result && !result->HasError()) {
        elog(INFO, "DuckDB enable_external_access: %s", result->GetValue(0, 0).ToString().c_str());
    }

    result = con->Query("SELECT current_setting('external_threads')");
    if (result && !result->HasError()) {
        elog(INFO, "DuckDB external_threads: %s", result->GetValue(0, 0).ToString().c_str());
    }

    result = con->Query("SELECT current_setting('temp_block_size')");
    if (result && !result->HasError()) {
        elog(INFO, "DuckDB temp_block_size: %s", result->GetValue(0, 0).ToString().c_str());
    }

    con->Query("SET enable_profiling = true");
    con->Query("SET profiling_mode = 'detailed'");

    std::string explain_query = "EXPLAIN " + query_string;
    auto explain_result = con->Query(explain_query);

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
duckdb_result_holder::duckdb_result_holder() = default;

duckdb_result_holder::~duckdb_result_holder() = default;

duckdb_result_holder::duckdb_result_holder(duckdb_result_holder&&) noexcept = default;

duckdb_result_holder& duckdb_result_holder::operator=(duckdb_result_holder&&) noexcept = default;

std::pair<size_t, size_t> duckdb_result_holder::get_chunk_and_offset(size_t global_row) const
{
    // DuckDB uses fixed chunk size (typically 2048)
    // We need to iterate through chunks to find the right one
    size_t accumulated_rows = 0;
    for (size_t chunk_idx = 0; chunk_idx < chunks.size(); ++chunk_idx) {
        size_t chunk_size = chunks[chunk_idx]->size();
        if (global_row < accumulated_rows + chunk_size) {
            return {chunk_idx, global_row - accumulated_rows};
        }
        accumulated_rows += chunk_size;
    }
    // Should not reach here if global_row < total_rows
    return {chunks.size() - 1, chunks.back()->size() - 1};
}

// Execute SQL query and return DuckDB results directly without conversion
duckdb_result_holder execute_sql_query_direct(const std::string& query_string)
{
    static std::unique_ptr<duckdb::Connection> con;
    if (con == nullptr || !pg::table_storage::instance().is_up_to_date()) {
        con = create_connection();
        auto& deeplake_tables = pg::table_storage::instance().get_tables();
        for (const auto& [table_id, table_data] : deeplake_tables) {
            register_table(con.get(), table_data.get_table_name(), table_id);
        }
        register_views(con.get());
        pg::table_storage::instance().set_up_to_date(true);
    }

    if (pg::explain_query_before_execute) {
        explain_query(con.get(), query_string);
    }

    // Execute the query
    std::unique_ptr<duckdb::QueryResult> result;
    {
        pg::runtime_printer printer("DuckDB query execution");
        result = execute_query(con.get(), query_string);
        ASSERT(result != nullptr);
    }

    // Fetch all chunks without converting to dataset_view
    duckdb_result_holder holder;
    holder.query_result = std::move(result);
    holder.total_rows = 0;

    while (true) {
        auto chunk = holder.query_result->Fetch();
        if (!chunk || chunk->size() == 0) {
            break;
        }
        holder.total_rows += chunk->size();
        holder.chunks.push_back(std::move(chunk));
    }

    elog(DEBUG1, "DuckDB query returned %zu rows in %zu chunks", holder.total_rows, holder.chunks.size());
    return holder;
}

} // namespace pg
