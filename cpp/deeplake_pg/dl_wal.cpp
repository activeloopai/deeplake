#include "dl_wal.hpp"

#include <codecs/compression.hpp>
#include <deeplake_api/catalog_table.hpp>
#include <deeplake_api/dataset.hpp>
#include <deeplake_core/type.hpp>
#include <nd/adapt.hpp>
#include <nd/array.hpp>
#include <nd/dtype.hpp>
#include <nd/type.hpp>

#include <algorithm>
#include <chrono>
#include <unordered_map>
#include <unistd.h>

extern "C" {
#include <postgres.h>
#include <miscadmin.h>
#include <utils/elog.h>
#include <utils/guc.h>
}

namespace pg::dl_wal {

namespace {

constexpr const char* k_catalog_dir = "__deeplake_catalog";
constexpr const char* k_databases_name = "databases";
constexpr const char* k_ddl_log_name = "__wal_table";

// Shared (cluster-wide) path: {root}/__deeplake_catalog/{name}
std::string join_path(const std::string& root, const std::string& name)
{
    if (!root.empty() && root.back() == '/') {
        return root + k_catalog_dir + "/" + name;
    }
    return root + "/" + k_catalog_dir + "/" + name;
}

// Per-database path: {root}/{db_name}/__deeplake_catalog/{name}
std::string join_db_path(const std::string& root, const std::string& db_name, const std::string& name)
{
    std::string base = root;
    if (!base.empty() && base.back() == '/') {
        base.pop_back();
    }
    return base + "/" + db_name + "/" + k_catalog_dir + "/" + name;
}

int64_t now_ms()
{
    using namespace std::chrono;
    return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}

// Open a shared (cluster-wide) catalog table
std::shared_ptr<deeplake_api::catalog_table>
open_catalog_table(const std::string& root_path, const std::string& name, icm::string_map<> creds)
{
    const auto path = join_path(root_path, name);
    return deeplake_api::open_catalog_table(path, std::move(creds)).get_future().get();
}

std::shared_ptr<deeplake_api::dataset>
open_or_create_ddl_dataset(const std::string& root_path, const std::string& db_name, icm::string_map<> creds)
{
    const auto path = join_db_path(root_path, db_name, k_ddl_log_name);
    std::shared_ptr<deeplake_api::dataset> ds;
    bool exists = false;
    try {
        exists = deeplake_api::exists(path, icm::string_map<>(creds)).get_future().get();
    } catch (...) {
        exists = false;
    }
    if (exists) {
        ds = deeplake_api::open(path, std::move(creds)).get_future().get();
    } else {
        ds = deeplake_api::create(path, std::move(creds)).get_future().get();
    }

    bool schema_changed = false;
    auto ensure_column = [&](const char* name, const deeplake_core::type& type) {
        try {
            (void)ds->get_column(name);
        } catch (...) {
            ds->add_column(name, type);
            schema_changed = true;
        }
    };

    ensure_column("seq", deeplake_core::type::generic(nd::type::scalar(nd::dtype::int64)));
    ensure_column("origin_instance_id", deeplake_core::type::text(codecs::compression::null));
    ensure_column("search_path", deeplake_core::type::text(codecs::compression::null));
    ensure_column("command_tag", deeplake_core::type::text(codecs::compression::null));
    ensure_column("object_identity", deeplake_core::type::text(codecs::compression::null));
    ensure_column("ddl_sql", deeplake_core::type::text(codecs::compression::null));
    ensure_column("timestamp", deeplake_core::type::generic(nd::type::scalar(nd::dtype::int64)));

    if (schema_changed) {
        ds->commit().get_future().get();
    }
    return ds;
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

deeplake_api::catalog_table_schema make_databases_schema()
{
    deeplake_api::catalog_table_schema schema;
    schema.add("db_name", deeplake_core::type::text(codecs::compression::null))
        .add("owner", deeplake_core::type::text(codecs::compression::null))
        .add("encoding", deeplake_core::type::text(codecs::compression::null))
        .add("lc_collate", deeplake_core::type::text(codecs::compression::null))
        .add("lc_ctype", deeplake_core::type::text(codecs::compression::null))
        .add("template_db", deeplake_core::type::text(codecs::compression::null))
        .add("state", deeplake_core::type::text(codecs::compression::null))
        .add("updated_at", deeplake_core::type::generic(nd::type::scalar(nd::dtype::int64)))
        .set_primary_key("db_name");
    return schema;
}

} // namespace

int64_t next_ddl_seq()
{
    static thread_local int64_t counter = 0;
    static thread_local int64_t last_ms = 0;
    const int64_t ms = now_ms();
    if (ms == last_ms) {
        counter++;
    } else {
        counter = 0;
        last_ms = ms;
    }
    return ms * 1000 + counter;
}

std::string local_instance_id()
{
    char hostname[256] = {0};
    if (gethostname(hostname, sizeof(hostname) - 1) != 0) {
        strlcpy(hostname, "unknown_host", sizeof(hostname));
    }
    const char* port = GetConfigOption("port", true, false);
    const std::string data_dir = DataDir != nullptr ? std::string(DataDir) : std::string("unknown_data_dir");
    const std::string host_str(hostname);
    const std::string port_str = port != nullptr ? std::string(port) : std::string("unknown_port");
    return host_str + ":" + port_str + ":" + data_dir;
}

void ensure_catalog(const std::string& root_path, icm::string_map<> creds)
{
    if (root_path.empty()) {
        return;
    }
    const auto databases_path = join_path(root_path, k_databases_name);

    try {
        deeplake_api::open_or_create_catalog_table(databases_path, make_databases_schema(), std::move(creds))
            .get_future()
            .get();
    } catch (const std::exception& e) {
        elog(ERROR, "Failed to ensure shared catalog at %s: %s", root_path.c_str(), e.what());
    } catch (...) {
        elog(ERROR, "Failed to ensure shared catalog at %s: unknown error", root_path.c_str());
    }
}

void ensure_db_catalog(const std::string& root_path, const std::string& db_name, icm::string_map<> creds)
{
    if (root_path.empty() || db_name.empty()) {
        return;
    }

    try {
        (void)open_or_create_ddl_dataset(root_path, db_name, std::move(creds));
    } catch (const std::exception& e) {
        elog(ERROR, "Failed to ensure per-db catalog at %s/%s: %s", root_path.c_str(), db_name.c_str(), e.what());
    } catch (...) {
        elog(ERROR, "Failed to ensure per-db catalog at %s/%s: unknown error", root_path.c_str(), db_name.c_str());
    }
}

std::vector<database_meta> load_databases(const std::string& root_path, icm::string_map<> creds)
{
    std::vector<database_meta> out;
    try {
        auto table = open_catalog_table(root_path, k_databases_name, std::move(creds));
        if (!table) {
            return out;
        }
        auto snapshot = table->read().get_future().get();
        if (snapshot.row_count() == 0) {
            return out;
        }

        std::unordered_map<std::string, database_meta> latest;
        for (const auto& row : snapshot.rows()) {
            auto db_name_it = row.find("db_name");
            auto owner_it = row.find("owner");
            auto encoding_it = row.find("encoding");
            auto lc_collate_it = row.find("lc_collate");
            auto lc_ctype_it = row.find("lc_ctype");
            auto template_it = row.find("template_db");
            auto state_it = row.find("state");
            auto updated_it = row.find("updated_at");
            if (db_name_it == row.end() || state_it == row.end()) {
                continue;
            }

            database_meta meta;
            meta.db_name = deeplake_api::array_to_string(db_name_it->second);
            if (owner_it != row.end()) meta.owner = deeplake_api::array_to_string(owner_it->second);
            if (encoding_it != row.end()) meta.encoding = deeplake_api::array_to_string(encoding_it->second);
            if (lc_collate_it != row.end()) meta.lc_collate = deeplake_api::array_to_string(lc_collate_it->second);
            if (lc_ctype_it != row.end()) meta.lc_ctype = deeplake_api::array_to_string(lc_ctype_it->second);
            if (template_it != row.end()) meta.template_db = deeplake_api::array_to_string(template_it->second);
            meta.state = deeplake_api::array_to_string(state_it->second);
            if (updated_it != row.end()) {
                auto updated_vec = load_int64_vector(updated_it->second);
                meta.updated_at = updated_vec.empty() ? 0 : updated_vec.front();
            }

            auto it = latest.find(meta.db_name);
            if (it == latest.end() || it->second.updated_at <= meta.updated_at) {
                latest[meta.db_name] = std::move(meta);
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
        elog(WARNING, "Failed to load catalog databases: %s", e.what());
        return out;
    } catch (...) {
        elog(WARNING, "Failed to load catalog databases: unknown error");
        return out;
    }
}

void upsert_database(const std::string& root_path, icm::string_map<> creds, const database_meta& meta)
{
    auto table = open_catalog_table(root_path, k_databases_name, std::move(creds));
    icm::string_map<nd::array> row;
    row["db_name"] = nd::adapt(meta.db_name);
    row["owner"] = nd::adapt(meta.owner);
    row["encoding"] = nd::adapt(meta.encoding);
    row["lc_collate"] = nd::adapt(meta.lc_collate);
    row["lc_ctype"] = nd::adapt(meta.lc_ctype);
    row["template_db"] = nd::adapt(meta.template_db);
    row["state"] = nd::adapt(meta.state);
    row["updated_at"] = nd::adapt(meta.updated_at == 0 ? now_ms() : meta.updated_at);
    table->upsert(std::move(row)).get_future().get();
}

int64_t get_databases_version(const std::string& root_path, icm::string_map<> creds)
{
    try {
        auto table = open_catalog_table(root_path, k_databases_name, std::move(creds));
        if (!table) {
            return 0;
        }
        return static_cast<int64_t>(table->version().get_future().get());
    } catch (const std::exception& e) {
        elog(WARNING, "Failed to read databases catalog version: %s", e.what());
        return 0;
    } catch (...) {
        elog(WARNING, "Failed to read databases catalog version: unknown error");
        return 0;
    }
}

std::shared_ptr<deeplake_api::dataset>
open_ddl_log_table(const std::string& root_path, const std::string& db_name, icm::string_map<> creds)
{
    return open_or_create_ddl_dataset(root_path, db_name, std::move(creds));
}

void append_ddl_log(const std::string& root_path, const std::string& db_name, icm::string_map<> creds,
                    const ddl_log_entry& entry)
{
    auto ds = open_ddl_log_table(root_path, db_name, std::move(creds));
    ds->set_auto_commit_enabled(false).get_future().get();
    icm::string_map<nd::array> row;
    row["seq"] = nd::adapt(entry.seq);
    row["origin_instance_id"] = nd::adapt(entry.origin_instance_id);
    row["search_path"] = nd::adapt(entry.search_path);
    row["command_tag"] = nd::adapt(entry.command_tag);
    row["object_identity"] = nd::adapt(entry.object_identity);
    row["ddl_sql"] = nd::adapt(entry.ddl_sql);
    row["timestamp"] = nd::adapt(entry.timestamp == 0 ? now_ms() : entry.timestamp);
    ds->append_row(row).get_future().get();
    ds->commit().get_future().get();
}

std::vector<ddl_log_entry> load_ddl_log(const std::string& root_path, const std::string& db_name,
                                        icm::string_map<> creds, int64_t after_seq)
{
    std::vector<ddl_log_entry> out;
    try {
        auto ds = open_ddl_log_table(root_path, db_name, std::move(creds));
        if (!ds) {
            return out;
        }
        ds->refresh().get_future().get();
        const int64_t row_count = ds->num_rows();
        if (row_count == 0) {
            return out;
        }

        auto seq_arr = ds->get_column("seq").request_range(0, row_count, {}).get_future().get();
        auto origin_arr = ds->get_column("origin_instance_id").request_range(0, row_count, {}).get_future().get();
        auto search_path_arr = ds->get_column("search_path").request_range(0, row_count, {}).get_future().get();
        auto tag_arr = ds->get_column("command_tag").request_range(0, row_count, {}).get_future().get();
        auto object_arr = ds->get_column("object_identity").request_range(0, row_count, {}).get_future().get();
        auto sql_arr = ds->get_column("ddl_sql").request_range(0, row_count, {}).get_future().get();
        auto ts_arr = ds->get_column("timestamp").request_range(0, row_count, {}).get_future().get();

        auto seq_vec = load_int64_vector(seq_arr);
        auto ts_vec = load_int64_vector(ts_arr);
        for (int64_t i = 0; i < row_count; ++i) {
            ddl_log_entry entry;
            entry.seq = i < static_cast<int64_t>(seq_vec.size()) ? seq_vec[static_cast<size_t>(i)] : 0;
            if (entry.seq <= after_seq) {
                continue;
            }
            auto read_string = [](const nd::array& arr, int64_t idx) -> std::string {
                try {
                    auto sub = arr[idx];
                    auto bytes = sub.data();
                    return std::string(reinterpret_cast<const char*>(bytes.data()), bytes.size());
                } catch (...) {
                    try {
                        return std::string(arr.value<std::string_view>(idx));
                    } catch (...) {
                        return {};
                    }
                }
            };
            entry.origin_instance_id = read_string(origin_arr, i);
            entry.search_path = read_string(search_path_arr, i);
            entry.command_tag = read_string(tag_arr, i);
            entry.object_identity = read_string(object_arr, i);
            entry.ddl_sql = read_string(sql_arr, i);
            entry.timestamp = i < static_cast<int64_t>(ts_vec.size()) ? ts_vec[static_cast<size_t>(i)] : 0;
            out.push_back(std::move(entry));
        }
        std::sort(out.begin(), out.end(), [](const ddl_log_entry& a, const ddl_log_entry& b) {
            return a.seq < b.seq;
        });
        return out;
    } catch (const std::exception& e) {
        elog(WARNING, "Failed to load DDL log for db '%s': %s", db_name.c_str(), e.what());
        return out;
    } catch (...) {
        elog(WARNING, "Failed to load DDL log for db '%s': unknown error", db_name.c_str());
        return out;
    }
}

} // namespace pg::dl_wal
