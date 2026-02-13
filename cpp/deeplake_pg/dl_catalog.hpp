#pragma once

#include <icm/string_map.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace deeplake_api { class catalog_table; }

namespace pg::dl_catalog {

struct table_meta
{
    std::string table_id;
    std::string schema_name;
    std::string table_name;
    std::string dataset_path;
    std::string state;
    std::string db_name;
    int64_t updated_at = 0;
};

struct column_meta
{
    std::string table_id;
    std::string column_name;
    std::string pg_type;
    std::string dl_type_json;
    bool nullable = true;
    int32_t position = 0;
};

struct index_meta
{
    std::string table_id;
    std::string column_names;
    std::string index_type;
    int32_t order_type = 0;
};

struct schema_meta
{
    std::string schema_name;   // PK
    std::string owner;
    std::string state;         // "ready" or "dropping"
    int64_t updated_at = 0;
};

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

// Shared (cluster-wide) catalog: meta + databases
int64_t ensure_catalog(const std::string& root_path, icm::string_map<> creds);

// Per-database catalog: tables + columns + indexes + meta
int64_t ensure_db_catalog(const std::string& root_path, const std::string& db_name, icm::string_map<> creds);

// Per-database loaders (read from {root}/{db_name}/__deeplake_catalog/)
std::vector<table_meta> load_tables(const std::string& root_path, const std::string& db_name, icm::string_map<> creds);
std::vector<column_meta> load_columns(const std::string& root_path, const std::string& db_name, icm::string_map<> creds);
std::vector<index_meta> load_indexes(const std::string& root_path, const std::string& db_name, icm::string_map<> creds);

// Load tables and columns in parallel for better performance
std::pair<std::vector<table_meta>, std::vector<column_meta>>
load_tables_and_columns(const std::string& root_path, const std::string& db_name, icm::string_map<> creds);

// Per-database schema catalog
std::vector<schema_meta> load_schemas(const std::string& root_path, const std::string& db_name, icm::string_map<> creds);
void upsert_schema(const std::string& root_path, const std::string& db_name, icm::string_map<> creds, const schema_meta& meta);

// Per-database upserts (write to {root}/{db_name}/__deeplake_catalog/)
void upsert_table(const std::string& root_path, const std::string& db_name, icm::string_map<> creds, const table_meta& meta);
void upsert_columns(const std::string& root_path, const std::string& db_name, icm::string_map<> creds, const std::vector<column_meta>& columns);
void upsert_indexes(const std::string& root_path, const std::string& db_name, icm::string_map<> creds, const std::vector<index_meta>& indexes);

// Shared (cluster-wide) database catalog
std::vector<database_meta> load_databases(const std::string& root_path, icm::string_map<> creds);
void upsert_database(const std::string& root_path, icm::string_map<> creds, const database_meta& meta);

// Global (shared) catalog version
int64_t get_catalog_version(const std::string& root_path, icm::string_map<> creds);
void bump_catalog_version(const std::string& root_path, icm::string_map<> creds);

// Per-database catalog version
int64_t get_db_catalog_version(const std::string& root_path, const std::string& db_name, icm::string_map<> creds);
void bump_db_catalog_version(const std::string& root_path, const std::string& db_name, icm::string_map<> creds);

// Open the per-database meta table handle (for parallel .version() calls in sync worker)
std::shared_ptr<deeplake_api::catalog_table>
open_db_meta_table(const std::string& root_path, const std::string& db_name, icm::string_map<> creds);

} // namespace pg::dl_catalog
