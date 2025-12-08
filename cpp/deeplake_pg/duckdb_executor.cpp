// Include DuckDB headers in implementation file only. Keep first to not conflict with PG headers
#include <duckdb.hpp>

#include "duckdb_executor.hpp"
#include "duckdb_deeplake_convert.hpp"
#include "duckdb_deeplake_scan.hpp"
#include "pg_deeplake.hpp"
#include "reporter.hpp"
#include "table_data.hpp"
#include "table_storage.hpp"
#include "utils.hpp"

#include <heimdall_common/array_column_view.hpp>
#include <heimdall_common/dataset_filtered_by_tensors.hpp>

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
        std::string create_view_sql = fmt::format(
            "CREATE OR REPLACE VIEW \"{}\".\"{}\" AS SELECT * FROM deeplake_scan(CAST({} AS UINTEGER))",
            schema_name, table_name.substr(dot_pos + 1), table_id
        );

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
        const auto set_search_path = fmt::format("SET search_path TO \"{}\"", pg::table_storage::instance().get_schema_name());
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

std::shared_ptr<deeplake_api::dataset_view> convert_result_to_dataset_generic(duckdb::QueryResult* result)
{
    std::vector<duckdb::vector<duckdb::Value>> column_data(result->ColumnCount());
    while (true) {
        auto data_chunk = result->Fetch();
        if (!data_chunk || data_chunk->size() == 0) {
            break;
        }

        ASSERT(data_chunk->ColumnCount() == result->ColumnCount());
        for (size_t i = 0; i < data_chunk->ColumnCount(); ++i) {
            auto& cd = column_data[i];
            for (size_t j = 0; j < data_chunk->size(); ++j) {
                cd.emplace_back(data_chunk->GetValue(i, j));
            }
        }
    }

    std::vector<heimdall::column_view_ptr> columns;
    for (size_t i = 0; i < result->ColumnCount(); ++i) {
        const auto& duckdb_type = result->types[i];
        auto type = pg::to_deeplake_type(duckdb_type);
        auto arr = pg::to_deeplake_value(duckdb::Value::LIST(duckdb_type, std::move(column_data[i])));
        columns.emplace_back(std::make_shared<heimdall_common::array_column>(result->names[i], type, std::move(arr)));
    }

    return heimdall_common::create_dataset_with_tensors(std::move(columns));
}

// consider handling date/timestamp types
// consider merge with above
std::shared_ptr<deeplake_api::dataset_view> convert_result_to_dataset_arithmetic(duckdb::QueryResult* result)
{
    size_t total_rows = 0;

    std::vector<std::unique_ptr<duckdb::DataChunk>> all_chunks;
    while (true) {
        auto data_chunk = result->Fetch();
        if (!data_chunk || data_chunk->size() == 0) {
            break;
        }
        total_rows += data_chunk->size();
        all_chunks.push_back(std::move(data_chunk));
    }

    // Consolidate all chunks into a single Vector per column for zero-copy optimization
    std::vector<std::shared_ptr<duckdb::Vector>> column_vectors;
    column_vectors.reserve(result->ColumnCount());
    for (size_t col_idx = 0; col_idx < result->ColumnCount(); ++col_idx) {
        const auto& duckdb_type = result->types[col_idx];

        // Create a single large Vector for the entire column
        auto consolidated_vector = std::make_shared<duckdb::Vector>(duckdb_type, total_rows);

        size_t offset = 0;
        for (const auto& chunk : all_chunks) {
            ASSERT(chunk->ColumnCount() == result->ColumnCount());
            const auto& src_vec = chunk->data[col_idx];
            const size_t chunk_size = chunk->size();

            // Copy chunk data into consolidated vector at offset
            duckdb::VectorOperations::Copy(src_vec, *consolidated_vector, chunk_size, 0, offset);
            offset += chunk_size;
        }

        ASSERT(offset == total_rows);
        column_vectors.emplace_back(std::move(consolidated_vector));
    }

    // Create columns using zero-copy access to DuckDB vector data
    std::vector<heimdall::column_view_ptr> columns;
    for (size_t i = 0; i < result->ColumnCount(); ++i) {
        const auto& duckdb_type = result->types[i];
        auto type = pg::to_deeplake_type(duckdb_type);

        // Create nd::array with zero-copy from DuckDB vector
        // The vector is kept alive by being stored in nd::array as owner
        auto arr = pg::to_deeplake_value(std::move(column_vectors[i]), total_rows);
        columns.emplace_back(std::make_shared<heimdall_common::array_column>(result->names[i], type, std::move(arr)));
    }

    return heimdall_common::create_dataset_with_tensors(std::move(columns));
}

std::shared_ptr<deeplake_api::dataset_view> convert_result_to_dataset(duckdb::QueryResult* result)
{
    bool all_types_are_arithmetic = true;
    for (size_t i = 0; i < result->ColumnCount(); ++i) {
        const auto& duckdb_type = result->types[i];
        // Trace hugeint(s) as non-arithmetic, as we dont have hugeint in deeplake
        if (!duckdb_type.IsNumeric() || duckdb::GetTypeIdSize(duckdb_type.InternalType()) > 8) {
            all_types_are_arithmetic = false;
            break;
        }
    }
    if (all_types_are_arithmetic) {
        return convert_result_to_dataset_arithmetic(result);
    }
    return convert_result_to_dataset_generic(result);
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

std::shared_ptr<deeplake_api::dataset_view> execute_sql_query(const std::string& query_string)
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

    // Convert result back to dataset format
    pg::runtime_printer printer("Convert DuckDB result to deeplake dataset");
    return convert_result_to_dataset(result.get());
}

} // pg namespace
