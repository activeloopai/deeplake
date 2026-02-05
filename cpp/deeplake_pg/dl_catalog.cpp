#include "dl_catalog.hpp"

#include <async/promise.hpp>
#include <codecs/compression.hpp>
#include <deeplake_api/catalog_table.hpp>
#include <deeplake_api/dataset.hpp>
#include <deeplake_core/type.hpp>
#include <nd/adapt.hpp>
#include <nd/array.hpp>
#include <nd/dtype.hpp>
#include <nd/type.hpp>

#include <algorithm>
#include <unordered_map>

extern "C" {
#include <postgres.h>
#include <utils/elog.h>
}

namespace pg::dl_catalog {

namespace {

constexpr const char* k_catalog_dir = "__deeplake_catalog";
constexpr const char* k_tables_name = "tables";
constexpr const char* k_columns_name = "columns";
constexpr const char* k_indexes_name = "indexes";
constexpr const char* k_meta_name = "meta";

std::string join_path(const std::string& root, const std::string& name)
{
    if (!root.empty() && root.back() == '/') {
        return root + k_catalog_dir + "/" + name;
    }
    return root + "/" + k_catalog_dir + "/" + name;
}

// Cache for catalog table handles to avoid repeated S3 opens
struct catalog_table_cache
{
    std::string root_path;
    std::shared_ptr<deeplake_api::catalog_table> meta_table;

    static catalog_table_cache& instance()
    {
        static thread_local catalog_table_cache cache;
        return cache;
    }

    std::shared_ptr<deeplake_api::catalog_table> get_meta_table(const std::string& path, icm::string_map<> creds)
    {
        if (path != root_path || !meta_table) {
            // Cache miss or path changed - open and cache
            root_path = path;
            const auto meta_path = join_path(path, k_meta_name);
            meta_table = deeplake_api::open_catalog_table(meta_path, std::move(creds)).get_future().get();
        }
        return meta_table;
    }

    void invalidate()
    {
        root_path.clear();
        meta_table.reset();
    }
};

std::shared_ptr<deeplake_api::catalog_table>
open_or_create_table(const std::string& path, deeplake_api::catalog_table_schema schema, icm::string_map<> creds)
{
    return deeplake_api::open_or_create_catalog_table(path, schema, std::move(creds)).get_future().get();
}

int64_t now_ms()
{
    using namespace std::chrono;
    return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}

std::shared_ptr<deeplake_api::catalog_table>
open_catalog_table(const std::string& root_path, const std::string& name, icm::string_map<> creds)
{
    const auto path = join_path(root_path, name);
    return deeplake_api::open_catalog_table(path, std::move(creds)).get_future().get();
}

template <typename T>
std::vector<T> load_vector(const nd::array& arr)
{
    std::vector<T> out;
    out.reserve(static_cast<size_t>(arr.volume()));
    for (int64_t i = 0; i < arr.volume(); ++i) {
        out.push_back(arr.value<T>(i));
    }
    return out;
}

std::vector<int64_t> load_int64_vector(const nd::array& arr)
{
    std::vector<int64_t> out;
    out.reserve(static_cast<size_t>(arr.volume()));
    bool is_numeric = false;
    try {
        is_numeric = nd::dtype_is_numeric(arr.dtype());
    } catch (...) {
        is_numeric = false;
    }
    if (is_numeric) {
        try {
            for (int64_t i = 0; i < arr.volume(); ++i) {
                out.push_back(arr.value<int64_t>(i));
            }
            return out;
        } catch (...) {
            out.clear();
        }
    }
    for (int64_t i = 0; i < arr.volume(); ++i) {
        auto v = arr.value<std::string_view>(i);
        try {
            out.push_back(std::stoll(std::string(v)));
        } catch (...) {
            out.push_back(0);
        }
    }
    return out;
}

} // namespace

void ensure_catalog(const std::string& root_path, icm::string_map<> creds)
{
    const auto tables_path = join_path(root_path, k_tables_name);
    const auto columns_path = join_path(root_path, k_columns_name);
    const auto indexes_path = join_path(root_path, k_indexes_name);
    const auto meta_path = join_path(root_path, k_meta_name);

    // Build schemas for all catalog tables
    deeplake_api::catalog_table_schema tables_schema;
    tables_schema.add("table_id", deeplake_core::type::text(codecs::compression::null))
        .add("schema_name", deeplake_core::type::text(codecs::compression::null))
        .add("table_name", deeplake_core::type::text(codecs::compression::null))
        .add("dataset_path", deeplake_core::type::text(codecs::compression::null))
        .add("state", deeplake_core::type::text(codecs::compression::null))
        .add("updated_at", deeplake_core::type::generic(nd::type::scalar(nd::dtype::int64)))
        .set_primary_key("table_id");

    deeplake_api::catalog_table_schema columns_schema;
    columns_schema.add("column_id", deeplake_core::type::text(codecs::compression::null))
        .add("table_id", deeplake_core::type::text(codecs::compression::null))
        .add("column_name", deeplake_core::type::text(codecs::compression::null))
        .add("pg_type", deeplake_core::type::text(codecs::compression::null))
        .add("dl_type_json", deeplake_core::type::text(codecs::compression::null))
        .add("nullable", deeplake_core::type::generic(nd::type::scalar(nd::dtype::boolean)))
        .add("position", deeplake_core::type::generic(nd::type::scalar(nd::dtype::int32)))
        .set_primary_key("column_id");

    deeplake_api::catalog_table_schema indexes_schema;
    indexes_schema.add("table_id", deeplake_core::type::text(codecs::compression::null))
        .add("column_names", deeplake_core::type::text(codecs::compression::null))
        .add("index_type", deeplake_core::type::text(codecs::compression::null))
        .add("order_type", deeplake_core::type::generic(nd::type::scalar(nd::dtype::int32)))
        .set_primary_key("table_id");

    deeplake_api::catalog_table_schema meta_schema;
    meta_schema.add("catalog_version", deeplake_core::type::generic(nd::type::scalar(nd::dtype::int64)))
        .add("updated_at", deeplake_core::type::generic(nd::type::scalar(nd::dtype::int64)))
        .set_primary_key("catalog_version");

    // Launch all 4 open_or_create operations in parallel
    std::vector<async::promise<std::shared_ptr<deeplake_api::catalog_table>>> promises;
    promises.reserve(4);
    promises.push_back(
        deeplake_api::open_or_create_catalog_table(tables_path, std::move(tables_schema), icm::string_map<>(creds)));
    promises.push_back(
        deeplake_api::open_or_create_catalog_table(columns_path, std::move(columns_schema), icm::string_map<>(creds)));
    promises.push_back(
        deeplake_api::open_or_create_catalog_table(indexes_path, std::move(indexes_schema), icm::string_map<>(creds)));
    promises.push_back(
        deeplake_api::open_or_create_catalog_table(meta_path, std::move(meta_schema), icm::string_map<>(creds)));

    // Wait for all to complete
    auto results = async::combine(std::move(promises)).get_future().get();

    // Initialize meta table if empty (index 3 is meta)
    auto& meta_table = results[3];
    if (meta_table) {
        auto snapshot = meta_table->read().get_future().get();
        if (snapshot.row_count() == 0) {
            icm::string_map<nd::array> row;
            row["catalog_version"] = nd::adapt(static_cast<int64_t>(1));
            row["updated_at"] = nd::adapt(now_ms());
            meta_table->insert(std::move(row)).get_future().get();
        }
    }
}

std::vector<table_meta> load_tables(const std::string& root_path, icm::string_map<> creds)
{
    std::vector<table_meta> out;
    try {
        auto table = open_catalog_table(root_path, k_tables_name, std::move(creds));
        if (!table) {
            return out;
        }
        auto snapshot = table->read().get_future().get();
        if (snapshot.row_count() == 0) {
            return out;
        }

        std::unordered_map<std::string, table_meta> latest;
        for (const auto& row : snapshot.rows()) {
            auto table_id_it = row.find("table_id");
            auto schema_it = row.find("schema_name");
            auto table_it = row.find("table_name");
            auto path_it = row.find("dataset_path");
            auto state_it = row.find("state");
            auto updated_it = row.find("updated_at");
            if (table_id_it == row.end() || schema_it == row.end() || table_it == row.end() || path_it == row.end() ||
                state_it == row.end() || updated_it == row.end()) {
                continue;
            }

            table_meta meta;
            meta.table_id = deeplake_api::array_to_string(table_id_it->second);
            meta.schema_name = deeplake_api::array_to_string(schema_it->second);
            meta.table_name = deeplake_api::array_to_string(table_it->second);
            meta.dataset_path = deeplake_api::array_to_string(path_it->second);
            meta.state = deeplake_api::array_to_string(state_it->second);
            auto updated_vec = load_int64_vector(updated_it->second);
            meta.updated_at = updated_vec.empty() ? 0 : updated_vec.front();

            auto it = latest.find(meta.table_id);
            if (it == latest.end() || it->second.updated_at <= meta.updated_at) {
                latest[meta.table_id] = std::move(meta);
            }
        }

        out.reserve(latest.size());
        for (auto& [_, meta] : latest) {
            if (meta.state == "ready") {
                out.push_back(std::move(meta));
            }
        }
        return out;
    } catch (const std::exception& e) {
        elog(WARNING, "Failed to load catalog tables: %s", e.what());
        return out;
    } catch (...) {
        elog(WARNING, "Failed to load catalog tables: unknown error");
        return out;
    }
}

std::vector<column_meta> load_columns(const std::string& root_path, icm::string_map<> creds)
{
    std::vector<column_meta> out;
    try {
        auto table = open_catalog_table(root_path, k_columns_name, std::move(creds));
        if (!table) {
            return out;
        }
        auto snapshot = table->read().get_future().get();
        if (snapshot.row_count() == 0) {
            return out;
        }

        for (const auto& row : snapshot.rows()) {
            auto table_id_it = row.find("table_id");
            auto column_name_it = row.find("column_name");
            auto pg_type_it = row.find("pg_type");
            auto dl_type_it = row.find("dl_type_json");
            auto nullable_it = row.find("nullable");
            auto position_it = row.find("position");

            if (table_id_it == row.end() || column_name_it == row.end() || pg_type_it == row.end()) {
                continue;
            }

            column_meta meta;
            meta.table_id = deeplake_api::array_to_string(table_id_it->second);
            meta.column_name = deeplake_api::array_to_string(column_name_it->second);
            meta.pg_type = deeplake_api::array_to_string(pg_type_it->second);
            if (dl_type_it != row.end()) {
                meta.dl_type_json = deeplake_api::array_to_string(dl_type_it->second);
            }
            if (nullable_it != row.end()) {
                try {
                    meta.nullable = nullable_it->second.value<bool>(0);
                } catch (...) {
                    meta.nullable = true;
                }
            }
            if (position_it != row.end()) {
                try {
                    meta.position = position_it->second.value<int32_t>(0);
                } catch (...) {
                    auto pos_vec = load_int64_vector(position_it->second);
                    meta.position = pos_vec.empty() ? 0 : static_cast<int32_t>(pos_vec.front());
                }
            }

            out.push_back(std::move(meta));
        }
        return out;
    } catch (const std::exception& e) {
        elog(WARNING, "Failed to load catalog columns: %s", e.what());
        return out;
    } catch (...) {
        elog(WARNING, "Failed to load catalog columns: unknown error");
        return out;
    }
}

std::vector<index_meta> load_indexes(const std::string&, icm::string_map<>)
{
    return {};
}

std::pair<std::vector<table_meta>, std::vector<column_meta>>
load_tables_and_columns(const std::string& root_path, icm::string_map<> creds)
{
    std::vector<table_meta> tables_out;
    std::vector<column_meta> columns_out;

    try {
        // Open both catalog tables in parallel
        auto tables_promise = deeplake_api::open_catalog_table(join_path(root_path, k_tables_name), icm::string_map<>(creds));
        auto columns_promise = deeplake_api::open_catalog_table(join_path(root_path, k_columns_name), icm::string_map<>(creds));

        std::vector<async::promise<std::shared_ptr<deeplake_api::catalog_table>>> open_promises;
        open_promises.push_back(std::move(tables_promise));
        open_promises.push_back(std::move(columns_promise));

        auto catalog_tables = async::combine(std::move(open_promises)).get_future().get();

        auto& tables_table = catalog_tables[0];
        auto& columns_table = catalog_tables[1];

        if (!tables_table || !columns_table) {
            return {tables_out, columns_out};
        }

        // Read both snapshots in parallel
        auto tables_read_promise = tables_table->read();
        auto columns_read_promise = columns_table->read();

        std::vector<async::promise<deeplake_api::catalog_table_snapshot>> read_promises;
        read_promises.push_back(std::move(tables_read_promise));
        read_promises.push_back(std::move(columns_read_promise));

        auto snapshots = async::combine(std::move(read_promises)).get_future().get();

        auto& tables_snapshot = snapshots[0];
        auto& columns_snapshot = snapshots[1];

        // Process tables
        if (tables_snapshot.row_count() > 0) {
            std::unordered_map<std::string, table_meta> latest;
            for (const auto& row : tables_snapshot.rows()) {
                auto table_id_it = row.find("table_id");
                auto schema_it = row.find("schema_name");
                auto table_it = row.find("table_name");
                auto path_it = row.find("dataset_path");
                auto state_it = row.find("state");
                auto updated_it = row.find("updated_at");
                if (table_id_it == row.end() || schema_it == row.end() || table_it == row.end() || path_it == row.end() ||
                    state_it == row.end() || updated_it == row.end()) {
                    continue;
                }

                table_meta meta;
                meta.table_id = deeplake_api::array_to_string(table_id_it->second);
                meta.schema_name = deeplake_api::array_to_string(schema_it->second);
                meta.table_name = deeplake_api::array_to_string(table_it->second);
                meta.dataset_path = deeplake_api::array_to_string(path_it->second);
                meta.state = deeplake_api::array_to_string(state_it->second);
                auto updated_vec = load_int64_vector(updated_it->second);
                meta.updated_at = updated_vec.empty() ? 0 : updated_vec.front();

                auto it = latest.find(meta.table_id);
                if (it == latest.end() || it->second.updated_at <= meta.updated_at) {
                    latest[meta.table_id] = std::move(meta);
                }
            }

            tables_out.reserve(latest.size());
            for (auto& [_, meta] : latest) {
                if (meta.state == "ready") {
                    tables_out.push_back(std::move(meta));
                }
            }
        }

        // Process columns
        if (columns_snapshot.row_count() > 0) {
            for (const auto& row : columns_snapshot.rows()) {
                auto table_id_it = row.find("table_id");
                auto column_name_it = row.find("column_name");
                auto pg_type_it = row.find("pg_type");
                auto dl_type_it = row.find("dl_type_json");
                auto nullable_it = row.find("nullable");
                auto position_it = row.find("position");

                if (table_id_it == row.end() || column_name_it == row.end() || pg_type_it == row.end()) {
                    continue;
                }

                column_meta meta;
                meta.table_id = deeplake_api::array_to_string(table_id_it->second);
                meta.column_name = deeplake_api::array_to_string(column_name_it->second);
                meta.pg_type = deeplake_api::array_to_string(pg_type_it->second);
                if (dl_type_it != row.end()) {
                    meta.dl_type_json = deeplake_api::array_to_string(dl_type_it->second);
                }
                if (nullable_it != row.end()) {
                    try {
                        meta.nullable = nullable_it->second.value<bool>(0);
                    } catch (...) {
                        meta.nullable = true;
                    }
                }
                if (position_it != row.end()) {
                    try {
                        meta.position = position_it->second.value<int32_t>(0);
                    } catch (...) {
                        auto pos_vec = load_int64_vector(position_it->second);
                        meta.position = pos_vec.empty() ? 0 : static_cast<int32_t>(pos_vec.front());
                    }
                }

                columns_out.push_back(std::move(meta));
            }
        }

        return {tables_out, columns_out};
    } catch (const std::exception& e) {
        elog(WARNING, "Failed to load catalog tables and columns: %s", e.what());
        return {tables_out, columns_out};
    } catch (...) {
        elog(WARNING, "Failed to load catalog tables and columns: unknown error");
        return {tables_out, columns_out};
    }
}

void upsert_table(const std::string& root_path, icm::string_map<> creds, const table_meta& meta)
{
    auto table = open_catalog_table(root_path, k_tables_name, std::move(creds));
    icm::string_map<nd::array> row;
    row["table_id"] = nd::adapt(meta.table_id);
    row["schema_name"] = nd::adapt(meta.schema_name);
    row["table_name"] = nd::adapt(meta.table_name);
    row["dataset_path"] = nd::adapt(meta.dataset_path);
    row["state"] = nd::adapt(meta.state);
    row["updated_at"] = nd::adapt(meta.updated_at == 0 ? now_ms() : meta.updated_at);
    table->upsert(std::move(row)).get_future().get();
}

void upsert_columns(const std::string& root_path, icm::string_map<> creds, const std::vector<column_meta>& columns)
{
    if (columns.empty()) {
        return;
    }
    auto table = open_catalog_table(root_path, k_columns_name, std::move(creds));
    std::vector<icm::string_map<nd::array>> rows;
    rows.reserve(columns.size());
    for (const auto& col : columns) {
        icm::string_map<nd::array> row;
        // column_id is the composite key: table_id:column_name
        row["column_id"] = nd::adapt(col.table_id + ":" + col.column_name);
        row["table_id"] = nd::adapt(col.table_id);
        row["column_name"] = nd::adapt(col.column_name);
        row["pg_type"] = nd::adapt(col.pg_type);
        row["dl_type_json"] = nd::adapt(col.dl_type_json);
        row["nullable"] = nd::adapt(col.nullable);
        row["position"] = nd::adapt(col.position);
        rows.push_back(std::move(row));
    }
    table->upsert_many(std::move(rows)).get_future().get();
}

int64_t get_catalog_version(const std::string& root_path, icm::string_map<> creds)
{
    try {
        // Use cached meta table handle to avoid repeated S3 opens
        auto table = catalog_table_cache::instance().get_meta_table(root_path, std::move(creds));
        if (!table) {
            return 0;
        }
        // Use version() for fast HEAD request instead of reading the whole table.
        // Returns a hash of the ETag which changes whenever the table is modified.
        return static_cast<int64_t>(table->version().get_future().get());
    } catch (const std::exception& e) {
        elog(WARNING, "Failed to read catalog version: %s", e.what());
        catalog_table_cache::instance().invalidate();
        return 0;
    } catch (...) {
        elog(WARNING, "Failed to read catalog version: unknown error");
        catalog_table_cache::instance().invalidate();
        return 0;
    }
}

void bump_catalog_version(const std::string& root_path, icm::string_map<> creds)
{
    auto table = open_catalog_table(root_path, k_meta_name, std::move(creds));
    icm::string_map<nd::array> row;
    // Use a fixed key and upsert - the updated_at timestamp change will trigger
    // a new ETag, which is what get_catalog_version() now detects via version().
    row["catalog_version"] = nd::adapt(static_cast<int64_t>(1));
    row["updated_at"] = nd::adapt(now_ms());
    table->upsert(std::move(row)).get_future().get();
}

} // namespace pg::dl_catalog
