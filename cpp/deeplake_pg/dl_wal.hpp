#pragma once

#include <icm/string_map.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace deeplake_api {
class dataset;
}

namespace pg::dl_wal {

struct database_meta
{
    std::string db_name;       // PK
    std::string owner;
    std::string encoding;
    std::string lc_collate;
    std::string lc_ctype;
    std::string template_db;
    std::string state;         // "ready" or "dropping"
    int64_t updated_at = 0;
};

struct ddl_log_entry
{
    int64_t seq = 0;               // Primary key
    std::string origin_instance_id;
    std::string search_path;
    std::string command_tag;
    std::string object_identity;
    std::string ddl_sql;
    int64_t timestamp = 0;
};

// Shared (cluster-wide) catalog: databases catalog_table
void ensure_catalog(const std::string& root_path, icm::string_map<> creds);

// Per-database catalog: __wal_table dataset
void ensure_db_catalog(const std::string& root_path, const std::string& db_name, icm::string_map<> creds);

// Shared (cluster-wide) database catalog
std::vector<database_meta> load_databases(const std::string& root_path, icm::string_map<> creds);
void upsert_database(const std::string& root_path, icm::string_map<> creds, const database_meta& meta);

// Global version check via databases catalog_table
int64_t get_databases_version(const std::string& root_path, icm::string_map<> creds);

std::shared_ptr<deeplake_api::dataset>
open_ddl_log_table(const std::string& root_path, const std::string& db_name, icm::string_map<> creds);

void append_ddl_log(const std::string& root_path, const std::string& db_name, icm::string_map<> creds,
                    const ddl_log_entry& entry);

std::vector<ddl_log_entry> load_ddl_log(const std::string& root_path, const std::string& db_name,
                                        icm::string_map<> creds, int64_t after_seq = 0);

int64_t next_ddl_seq();

// Unique identifier for this PostgreSQL instance: "hostname:port:datadir"
std::string local_instance_id();

} // namespace pg::dl_wal
