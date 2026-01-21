#pragma once

#include <icm/string_map.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace pg::dl_catalog {

struct table_meta
{
    std::string table_id;
    std::string schema_name;
    std::string table_name;
    std::string dataset_path;
    std::string state;
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

void ensure_catalog(const std::string& root_path, icm::string_map<> creds);

std::vector<table_meta> load_tables(const std::string& root_path, icm::string_map<> creds);
std::vector<column_meta> load_columns(const std::string& root_path, icm::string_map<> creds);
std::vector<index_meta> load_indexes(const std::string& root_path, icm::string_map<> creds);

void upsert_table(const std::string& root_path, icm::string_map<> creds, const table_meta& meta);

int64_t get_catalog_version(const std::string& root_path, icm::string_map<> creds);
void bump_catalog_version(const std::string& root_path, icm::string_map<> creds);

} // namespace pg::dl_catalog
