#include "dl_catalog.hpp"

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

#include <chrono>

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

std::shared_ptr<deeplake_api::catalog_table> open_or_create_table(
    const std::string& path,
    deeplake_api::catalog_table_schema schema,
    icm::string_map<> creds)
{
    return deeplake_api::open_or_create_catalog_table(path, schema, std::move(creds)).get_future().get();
}

int64_t now_ms()
{
    using namespace std::chrono;
    return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}

std::shared_ptr<deeplake_api::catalog_table> open_catalog_table(const std::string& root_path,
                                                                const std::string& name,
                                                                icm::string_map<> creds)
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

    auto ensure_table = [&](const std::string& path,
                            deeplake_api::catalog_table_schema schema) -> std::shared_ptr<deeplake_api::catalog_table> {
        bool exists = false;
        try {
            exists = deeplake_api::exists(path, icm::string_map<>(creds)).get_future().get();
        } catch (...) {
            exists = false;
        }
        if (exists) {
            bool is_catalog = false;
            try {
                is_catalog = deeplake_api::is_catalog_table(path, icm::string_map<>(creds)).get_future().get();
            } catch (...) {
                is_catalog = false;
            }
            if (!is_catalog) {
                elog(WARNING,
                     "Existing catalog path %s is not a catalog table. Recreating catalog table.",
                     path.c_str());
                try {
                    deeplake_api::delete_dataset(path, icm::string_map<>(creds)).get_future().get();
                } catch (...) {
                    elog(WARNING, "Failed to delete legacy dataset at %s", path.c_str());
                }
            }
        }
        try {
            return open_or_create_table(path, std::move(schema), icm::string_map<>(creds));
        } catch (const std::exception& e) {
            elog(ERROR, "Failed to open or create catalog table at %s: %s", path.c_str(), e.what());
        }
        return {};
    };

    {
        deeplake_api::catalog_table_schema schema;
        schema.add("table_id", deeplake_core::type::text(codecs::compression::null))
            .add("schema_name", deeplake_core::type::text(codecs::compression::null))
            .add("table_name", deeplake_core::type::text(codecs::compression::null))
            .add("dataset_path", deeplake_core::type::text(codecs::compression::null))
            .add("state", deeplake_core::type::text(codecs::compression::null))
            .add("updated_at", deeplake_core::type::generic(nd::type::scalar(nd::dtype::int64)))
            .set_primary_key("table_id");
        ensure_table(tables_path, std::move(schema));
    }

    {
        deeplake_api::catalog_table_schema schema;
        schema.add("table_id", deeplake_core::type::text(codecs::compression::null))
            .add("column_name", deeplake_core::type::text(codecs::compression::null))
            .add("pg_type", deeplake_core::type::text(codecs::compression::null))
            .add("dl_type_json", deeplake_core::type::text(codecs::compression::null))
            .add("nullable", deeplake_core::type::generic(nd::type::scalar(nd::dtype::boolean)))
            .add("position", deeplake_core::type::generic(nd::type::scalar(nd::dtype::int32)))
            .set_primary_key("table_id");
        ensure_table(columns_path, std::move(schema));
    }

    {
        deeplake_api::catalog_table_schema schema;
        schema.add("table_id", deeplake_core::type::text(codecs::compression::null))
            .add("column_names", deeplake_core::type::text(codecs::compression::null))
            .add("index_type", deeplake_core::type::text(codecs::compression::null))
            .add("order_type", deeplake_core::type::generic(nd::type::scalar(nd::dtype::int32)))
            .set_primary_key("table_id");
        ensure_table(indexes_path, std::move(schema));
    }

    deeplake_api::catalog_table_schema meta_schema;
    meta_schema.add("catalog_version", deeplake_core::type::generic(nd::type::scalar(nd::dtype::int64)))
        .add("updated_at", deeplake_core::type::generic(nd::type::scalar(nd::dtype::int64)))
        .set_primary_key("catalog_version");
    auto meta_table = ensure_table(meta_path, std::move(meta_schema));
    auto snapshot = meta_table->read().get_future().get();
    if (snapshot.row_count() == 0) {
        icm::string_map<nd::array> row;
        row["catalog_version"] = nd::adapt(static_cast<int64_t>(1));
        row["updated_at"] = nd::adapt(now_ms());
        meta_table->insert(std::move(row)).get_future().get();
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
            if (table_id_it == row.end() || schema_it == row.end() || table_it == row.end() ||
                path_it == row.end() || state_it == row.end() || updated_it == row.end()) {
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

std::vector<column_meta> load_columns(const std::string&, icm::string_map<>)
{
    return {};
}

std::vector<index_meta> load_indexes(const std::string&, icm::string_map<>)
{
    return {};
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

int64_t get_catalog_version(const std::string& root_path, icm::string_map<> creds)
{
    try {
        auto table = open_catalog_table(root_path, k_meta_name, std::move(creds));
        if (!table) {
            return 0;
        }
        auto snapshot = table->read().get_future().get();
        if (snapshot.row_count() == 0) {
            return 0;
        }
        int64_t max_version = 0;
        for (const auto& row : snapshot.rows()) {
            auto it = row.find("catalog_version");
            if (it == row.end()) {
                continue;
            }
            auto values = load_int64_vector(it->second);
            if (!values.empty()) {
                max_version = std::max(max_version, values.front());
            }
        }
        return max_version;
    } catch (const std::exception& e) {
        elog(WARNING, "Failed to read catalog version: %s", e.what());
        return 0;
    } catch (...) {
        elog(WARNING, "Failed to read catalog version: unknown error");
        return 0;
    }
}

void bump_catalog_version(const std::string& root_path, icm::string_map<> creds)
{
    auto table = open_catalog_table(root_path, k_meta_name, std::move(creds));
    int64_t version = get_catalog_version(root_path, creds);
    icm::string_map<nd::array> row;
    row["catalog_version"] = nd::adapt(version + 1);
    row["updated_at"] = nd::adapt(now_ms());
    table->insert(std::move(row)).get_future().get();
}

} // namespace pg::dl_catalog
