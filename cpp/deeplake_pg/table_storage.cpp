#include "pg_deeplake.hpp"

#ifdef __cplusplus
extern "C" {
#endif

#include <postgres.h>

#include <access/heapam.h>
#include <access/htup_details.h>
#include <access/parallel.h>
#include <access/xact.h>
#include <catalog/namespace.h>
#include <catalog/pg_type.h>
#include <commands/dbcommands.h>
#include <executor/spi.h>
#include <miscadmin.h>
#include <nodes/makefuncs.h>
#include <nodes/parsenodes.h>
#include <utils/builtins.h>
#include <utils/elog.h>
#include <utils/errcodes.h>
#include <utils/guc.h>
#include <utils/lsyscache.h>
#include <utils/rel.h>
#include <utils/snapmgr.h>
#include <utils/syscache.h>

#ifdef __cplusplus
}
#endif

#include "table_storage.hpp"

#include "dl_wal.hpp"
#include "exceptions.hpp"
#include "logger.hpp"
#include "memory_tracker.hpp"
#include "nd_utils.hpp"
#include "pg_version_compat.h"
#include "table_ddl_lock.hpp"
#include "table_scan.hpp"
#include "utils.hpp"
#include <storage/exceptions.hpp>

#include <icm/json.hpp>
#include <icm/string_map.hpp>
#include <nd/none.hpp>
#include <unordered_set>

#include <algorithm>
#include <vector>

namespace {

std::string get_qualified_table_name(Relation rel)
{
    Oid nspid = RelationGetNamespace(rel);
    char* nspname = get_namespace_name(nspid);
    std::string qualified_name = std::string(nspname ? nspname : "public") + "." + RelationGetRelationName(rel);
    if (nspname) {
        pfree(nspname);
    }
    return qualified_name;
}

// Helper function to split schema.table_name
std::pair<std::string, std::string> split_table_name(const std::string& full_name)
{
    auto dot_pos = full_name.find('.');
    if (dot_pos == std::string::npos) {
        return {"public", full_name}; // Default to public schema if not specified
    }
    return {full_name.substr(0, dot_pos), full_name.substr(dot_pos + 1)};
}

// Helper function to get default value for NULL numeric/scalar columns
nd::array get_default_value_for_null(Oid base_typeid)
{
    switch (base_typeid) {
    case INT2OID:
        return nd::adapt(static_cast<int16_t>(0));
    case INT4OID:
    case DATEOID:
        return nd::adapt(static_cast<int32_t>(0));
    case TIMEOID:
    case TIMESTAMPOID:
    case TIMESTAMPTZOID:
    case INT8OID:
        return nd::adapt(static_cast<int64_t>(0));
    case FLOAT4OID:
        return nd::adapt(static_cast<float>(0.0));
    case NUMERICOID:
    case FLOAT8OID:
        return nd::adapt(static_cast<double>(0.0));
    case BOOLOID:
        return nd::adapt(false);
    default:
        // For non-numeric types, use nd::none
        return nd::none(nd::dtype::unknown, 0);
    }
}

void convert_pg_to_nd(const pg::table_data& table_data,
                      const std::vector<Datum>& values,
                      const std::vector<uint8_t>& nulls,
                      int32_t t_len,
                      icm::string_map<nd::array>& row)
{
    TupleDesc tupdesc = table_data.get_tuple_descriptor();
    for (auto i = 0; i < table_data.num_columns(); ++i) {
        // Get the actual TupleDesc index for this logical column (handles dropped columns)
        const auto tupdesc_idx = table_data.get_tupdesc_index(i);
        Form_pg_attribute attr = TupleDescAttr(tupdesc, tupdesc_idx);
        if (attr == nullptr) {
            ereport(ERROR, (errcode(ERRCODE_INTERNAL_ERROR), errmsg("Invalid attribute at position %d", i)));
        }

        // Skip SERIAL columns as they are auto-generated
        if (attr->attidentity == 'a' || attr->attgenerated == ATTRIBUTE_GENERATED_STORED) {
            continue;
        }

        const auto column_name = table_data.get_atttypename(i);
        // Skip if column is not in the input tuple (use tupdesc_idx for values/nulls arrays)
        if (tupdesc_idx >= t_len || nulls[tupdesc_idx] == 1) {
            // For numeric/scalar columns with NULL value, assign default (0) value
            row[column_name] = ::get_default_value_for_null(table_data.get_base_atttypid(i));
            continue;
        }
        row[column_name] =
            pg::utils::datum_to_nd(values[tupdesc_idx], table_data.get_base_atttypid(i), table_data.get_atttypmod(i));
    }
}

std::string get_current_database_name()
{
    const char* dbname = get_database_name(MyDatabaseId);
    if (!dbname) return "postgres";
    std::string result(dbname);
    pfree(const_cast<char*>(dbname));
    return result;
}

} // unnamed namespace

namespace pg {

// Initialize static members
char* session_credentials::creds_guc_string = nullptr;
char* session_credentials::root_path_guc_string = nullptr;

icm::string_map<> session_credentials::get_credentials()
{
    icm::string_map<> creds_map;

    if (creds_guc_string != nullptr && std::strlen(creds_guc_string) > 0) {
        try {
            auto creds_json = icm::json::parse(creds_guc_string);
            for (auto it = creds_json.begin(); it != creds_json.end(); ++it) {
                std::string key_str(it.key());
                if (it.value().is_string()) {
                    creds_map[key_str] = it.value().get<std::string>();
                } else {
                    elog(WARNING, "Credential value for key '%s' is not a string, skipping", key_str.c_str());
                }
            }
        } catch (const std::exception& e) {
            elog(WARNING, "Failed to parse deeplake.creds: %s. Using environment variables.", e.what());
        }
    }

    return creds_map;
}

std::string session_credentials::get_root_path()
{
    // Environment variable takes priority over per-session GUC
    auto root = base::getenv<std::string>("DEEPLAKE_ROOT_PATH", "");
    if (!root.empty()) {
        return root;
    }
    if (root_path_guc_string != nullptr && std::strlen(root_path_guc_string) > 0) {
        return std::string(root_path_guc_string);
    }
    return "";
}

void session_credentials::initialize_guc()
{
    DefineCustomStringVariable(
        "deeplake.creds",
        "JSON string containing storage credentials for DeepLake datasets",
        "Credentials are used for all dataset operations in the current session. "
        "Example: SET deeplake.creds = '{\"aws_access_key_id\": \"...\", \"aws_secret_access_key\": \"...\"}';",
        &creds_guc_string, // linked C variable
        "",                // default value (empty)
        PGC_USERSET,       // context - can be set by any user
        0,                 // flags
        nullptr,           // check_hook
        nullptr,           // assign_hook
        nullptr            // show_hook
    );

    DefineCustomStringVariable(
        "deeplake.root_path",
        "Root path for DeepLake datasets",
        "Defines the root directory where datasets will be created when no explicit path is provided. "
        "Supports local paths and cloud storage (s3://, gcs://, azure://). "
        "Example: SET deeplake.root_path = 's3://my-bucket/datasets';",
        &root_path_guc_string, // linked C variable
        "",                    // default value (empty)
        PGC_USERSET,           // context - can be set by any user
        0,                     // flags
        nullptr,               // check_hook
        nullptr,               // assign_hook
        nullptr                // show_hook
    );
}

void table_storage::save_table_metadata(const pg::table_data& table_data)
{
    const std::string& table_name = table_data.get_table_name();
    const std::string ds_path = table_data.get_dataset_path().url();
    pg::utils::memory_context_switcher context_switcher;
    pg::utils::pg_try([&]() {
        StringInfoData buf;
        initStringInfo(&buf);

        appendStringInfo(&buf,
                         "INSERT INTO public.pg_deeplake_tables (table_oid, table_name, ds_path) "
                         "VALUES (%u, %s, %s) "
                         "ON CONFLICT DO NOTHING",
                         table_data.get_table_oid(),
                         quote_literal_cstr(table_name.c_str()),
                         quote_literal_cstr(ds_path.c_str()));

        pg::utils::spi_connector connector;
        if (SPI_execute(buf.data, false, 0) != SPI_OK_INSERT) {
            throw pg::exception("Failed to save table metadata");
        }
        return true;
    });
}

void table_storage::load_table_metadata()
{
    // Prevent recursion from SQL queries triggering hooks that call back here
    static thread_local bool loading_in_progress = false;
    if (loading_in_progress) {
        return;
    }
    loading_in_progress = true;
    struct guard { ~guard() { loading_in_progress = false; } } recursion_guard;

    const auto root_dir = []() {
        auto root = session_credentials::get_root_path();
        if (root.empty()) {
            root = pg::utils::get_deeplake_root_directory();
        }
        return root;
    }();
    auto creds = session_credentials::get_credentials();

    // Stateless sync via DDL WAL replay (only when enabled and root_dir is configured)
    if (pg::stateless_enabled && !root_dir.empty()) {
        const auto db_name = get_current_database_name();

        // Ensure both shared and per-database catalogs exist
        pg::dl_wal::ensure_catalog(root_dir, creds);
        pg::dl_wal::ensure_db_catalog(root_dir, db_name, creds);

        auto is_sync_replay_backend = []() {
            const char* app_name = GetConfigOption("application_name", true, false);
            return app_name != nullptr && strcmp(app_name, "pg_deeplake_sync") == 0;
        };
        if (!in_ddl_context() && !AmBackgroundWorkerProcess() && !is_sync_replay_backend()) {
            auto entries = pg::dl_wal::load_ddl_log(root_dir, db_name, creds, ddl_log_last_seq_);
            if (!entries.empty()) {
                elog(LOG, "pg_deeplake: DDL WAL replay: %zu entries to process for db '%s' (after seq %ld)",
                     entries.size(), db_name.c_str(), ddl_log_last_seq_);
            }
            for (const auto& entry : entries) {
                if (entry.seq > ddl_log_last_seq_) {
                    ddl_log_last_seq_ = entry.seq;
                }
                if (entry.origin_instance_id == pg::dl_wal::local_instance_id()) {
                    continue;
                }

                // Snapshot tables_ keys so we can roll back C++ state on failure
                std::vector<Oid> tables_before;
                tables_before.reserve(tables_.size());
                for (const auto& [oid, _] : tables_) {
                    tables_before.push_back(oid);
                }

                MemoryContext saved_context = CurrentMemoryContext;
                ResourceOwner saved_owner = CurrentResourceOwner;
                BeginInternalSubTransaction(nullptr);
                PG_TRY();
                {
                    set_catalog_only_create(true);
                    SPI_connect();
                    bool pushed_snapshot = false;
                    if (!ActiveSnapshotSet()) {
                        PushActiveSnapshot(GetTransactionSnapshot());
                        pushed_snapshot = true;
                    }
                    // Restore the original search_path so unqualified names resolve correctly
                    std::string saved_search_path;
                    if (!entry.search_path.empty()) {
                        const char* current_sp = GetConfigOption("search_path", true, false);
                        if (current_sp != nullptr) {
                            saved_search_path = current_sp;
                        }
                        StringInfoData sp_sql;
                        initStringInfo(&sp_sql);
                        appendStringInfo(&sp_sql,
                                         "SELECT pg_catalog.set_config('search_path', %s, true)",
                                         quote_literal_cstr(entry.search_path.c_str()));
                        SPI_execute(sp_sql.data, true, 0);
                        pfree(sp_sql.data);
                    }
                    SPI_execute(entry.ddl_sql.c_str(), false, 0);
                    // Restore the session's original search_path
                    if (!entry.search_path.empty()) {
                        StringInfoData restore_sql;
                        initStringInfo(&restore_sql);
                        appendStringInfo(&restore_sql,
                                         "SELECT pg_catalog.set_config('search_path', %s, true)",
                                         quote_literal_cstr(saved_search_path.c_str()));
                        SPI_execute(restore_sql.data, true, 0);
                        pfree(restore_sql.data);
                    }
                    if (pushed_snapshot) {
                        PopActiveSnapshot();
                    }
                    SPI_finish();
                    set_catalog_only_create(false);
                    ReleaseCurrentSubTransaction();
                }
                PG_CATCH();
                {
                    set_catalog_only_create(false);
                    MemoryContextSwitchTo(saved_context);
                    ErrorData* edata = CopyErrorData();
                    CurrentResourceOwner = saved_owner;
                    RollbackAndReleaseCurrentSubTransaction();
                    FlushErrorState();

                    // Remove any tables_ entries added during the failed replay,
                    // since the subtransaction rollback undid the catalog changes
                    // but the C++ map entries persist.
                    std::unordered_set<Oid> before_set(tables_before.begin(), tables_before.end());
                    for (auto it = tables_.begin(); it != tables_.end(); ) {
                        if (!before_set.contains(it->first)) {
                            it = tables_.erase(it);
                        } else {
                            ++it;
                        }
                    }

                    elog(WARNING, "pg_deeplake: DDL WAL replay failed (seq=%ld, tag=%s): %s (SQL: %.200s)",
                         entry.seq, entry.command_tag.c_str(),
                         edata->message ? edata->message : "unknown error",
                         entry.ddl_sql.c_str());
                    FreeErrorData(edata);
                }
                PG_END_TRY();
            }
        }
    }

    // Load from local pg_deeplake_tables
    if (tables_loaded_) {
        return;
    }
    tables_loaded_ = true;

    if (!pg::utils::check_table_exists("pg_deeplake_tables")) {
        elog(LOG, "pg_deeplake: pg_deeplake_tables does not exist, skipping local scan");
        return;
    }

    struct snapshot_guard
    {
        bool active = false;
        snapshot_guard()
        {
            if (!ActiveSnapshotSet()) {
                PushActiveSnapshot(GetTransactionSnapshot());
                active = true;
            }
        }
        ~snapshot_guard()
        {
            if (active) {
                PopActiveSnapshot();
            }
        }
    } guard;

    // Backward compatibility: Check if table_oid column exists
    // If not, drop and recreate the table with the correct schema
    if (!pg::utils::check_column_exists("pg_deeplake_tables", "table_oid")) {
        base::log_warning(base::log_channel::generic,
                          "Detected old schema for pg_deeplake_tables without table_oid column. "
                          "Dropping and recreating table to match current schema.");

        pg::utils::spi_connector connector;
        const char* drop_query = "DROP TABLE IF EXISTS public.pg_deeplake_tables CASCADE";
        if (SPI_execute(drop_query, false, 0) != SPI_OK_UTILITY) {
            base::log_warning(base::log_channel::generic, "Failed to drop old pg_deeplake_tables table");
        }

        const char* create_query = "CREATE TABLE public.pg_deeplake_tables ("
                                   "    id SERIAL PRIMARY KEY,"
                                   "    table_oid OID NOT NULL UNIQUE,"
                                   "    table_name NAME NOT NULL UNIQUE,"
                                   "    ds_path TEXT NOT NULL UNIQUE"
                                   ")";
        if (SPI_execute(create_query, false, 0) != SPI_OK_UTILITY) {
            base::log_warning(base::log_channel::generic, "Failed to create new pg_deeplake_tables table");
        }

        const char* grant_query = "GRANT SELECT, INSERT, UPDATE, DELETE ON public.pg_deeplake_tables TO PUBLIC";
        if (SPI_execute(grant_query, false, 0) != SPI_OK_UTILITY) {
            base::log_warning(base::log_channel::generic, "Failed to grant permissions on pg_deeplake_tables table");
        }

        // Table is now empty, so we can return early
        return;
    }

    const char* query = "SELECT table_oid, table_name, ds_path FROM public.pg_deeplake_tables";

    pg::utils::spi_connector connector;

    if (SPI_execute(query, true, 0) != SPI_OK_SELECT) {
        base::log_warning(base::log_channel::generic, "Failed to query table metadata");
        return;
    }

    const auto proc = SPI_processed;
    const bool res = (proc > 0 && SPI_tuptable != nullptr);
    if (!res) {
        return;
    }
    TupleDesc tupdesc = SPI_tuptable->tupdesc;
    SPITupleTable* tuptable = SPI_tuptable;

    // Get credentials from current session
    creds = session_credentials::get_credentials();

    std::vector<Oid> invalid_table_oids;
    for (auto i = 0; i < proc; ++i) {
        HeapTuple tuple = tuptable->vals[i];
        bool is_null = false;
        Oid relid = InvalidOid;
        Datum relid_datum = SPI_getbinval(tuple, tupdesc, 1, &is_null);
        if (!is_null) {
            relid = DatumGetUInt32(relid_datum);
        }
        const char* table_name = SPI_getvalue(tuple, tupdesc, 2);
        const char* ds_path = SPI_getvalue(tuple, tupdesc, 3);

        if (relid == InvalidOid || table_name == nullptr || ds_path == nullptr || tables_.contains(relid)) {
            continue;
        }
        try {
            // Get the relation and its tuple descriptor
            Relation rel = try_relation_open(relid, NoLock);
            if (rel == nullptr) {
                elog(WARNING, "Could not open relation for table %s", table_name);
                invalid_table_oids.push_back(relid);
                continue;
            }
            {
                pg::utils::memory_context_switcher context_switcher(TopMemoryContext);
                // Use the actual relation name from PostgreSQL catalog, not the cached metadata name
                // This ensures we have the current name even if the table was renamed
                std::string actual_table_name = get_qualified_table_name(rel);
                elog(DEBUG1,
                     "Loading table from metadata: cached_name=%s, actual_name=%s",
                     table_name,
                     actual_table_name.c_str());
                table_data td(
                    relid, actual_table_name, CreateTupleDescCopy(RelationGetDescr(rel)), std::string(ds_path), creds);
                auto it2status = tables_.emplace(relid, std::move(td));
                up_to_date_ = false;
                ASSERT(it2status.second);
            }
            relation_close(rel, NoLock);
        } catch (const base::exception& e) {
            ereport(ERROR, (errcode(ERRCODE_INTERNAL_ERROR), errmsg("%s", e.what())));
        }
    }
    for (Oid invalid_oid : invalid_table_oids) {
        auto query = fmt::format("DELETE FROM public.pg_deeplake_tables WHERE table_oid = {}", invalid_oid);
        if (SPI_execute(query.c_str(), false, 0) != SPI_OK_DELETE) {
            base::log_warning(
                base::log_channel::generic, "Failed to delete invalid table metadata for table_oid: {}", invalid_oid);
        }
    }
    load_views();
    load_schema_name();
}

void table_storage::load_views()
{
    if (!pg::utils::check_table_exists("pg_deeplake_views")) {
        return;
    }
    const char* view_query = "SELECT view_name, query_string FROM public.pg_deeplake_views";
    if (SPI_execute(view_query, true, 0) != SPI_OK_SELECT) {
        base::log_warning(base::log_channel::generic, "Failed to query view metadata");
        return;
    }

    const auto proc = SPI_processed;
    const bool res = (proc > 0 && SPI_tuptable != nullptr);
    if (!res) {
        return;
    }
    const auto tupdesc = SPI_tuptable->tupdesc;
    const auto tuptable = SPI_tuptable;

    for (auto i = 0; i < proc; ++i) {
        HeapTuple tuple = tuptable->vals[i];
        char* view_name = SPI_getvalue(tuple, tupdesc, 1);
        char* view_str = SPI_getvalue(tuple, tupdesc, 2);
        if (view_name == nullptr || view_str == nullptr) {
            continue;
        }
        Oid view_oid = RelnameGetRelid(view_name);
        if (OidIsValid(view_oid) && !views_.contains(view_oid)) {
            views_.emplace(view_oid, std::pair{view_name, view_str});
            up_to_date_ = false;
        }
    }
}

void table_storage::load_schema_name()
{
    const char* search_path = GetConfigOption("search_path", false, false);
    std::string schema_name = "public";
    std::string_view sv(search_path);

    // Check if search_path contains 'public'
    if (sv.find("public") == std::string_view::npos) {
        // If 'public' is not in search_path, take the first schema from comma-separated list
        auto comma_pos = sv.find(',');
        if (comma_pos != std::string_view::npos) {
            sv = sv.substr(0, comma_pos);
        }
        // Trim leading/trailing whitespace
        while (!sv.empty() && std::isspace(sv.front())) {
            sv.remove_prefix(1);
        }
        while (!sv.empty() && std::isspace(sv.back())) {
            sv.remove_suffix(1);
        }
        if (!sv.empty()) {
            schema_name = std::string(sv);
        }
    }
    schema_name_ = std::move(schema_name);
}

void table_storage::erase_table_metadata(const std::string& table_name)
{
    pg::utils::memory_context_switcher context_switcher;
    if (!pg::utils::check_table_exists("pg_deeplake_tables")) {
        return;
    }
    StringInfoData buf;
    initStringInfo(&buf);

    appendStringInfo(
        &buf, "DELETE FROM public.pg_deeplake_tables WHERE table_name = %s", quote_literal_cstr(table_name.c_str()));

    pg::utils::spi_connector connector;
    if (SPI_execute(buf.data, false, 0) != SPI_OK_DELETE) {
        ereport(ERROR, (errcode(ERRCODE_INTERNAL_ERROR), errmsg("Failed to erase table metadata")));
    }
}

void table_storage::create_table(const std::string& table_name, Oid table_id, TupleDesc tupdesc)
{
    // Acquire global DDL lock to prevent concurrent CREATE/DROP TABLE operations (also needed for deeplake)
    pg::table_ddl_lock_guard ddl_lock;

    pg::utils::memory_context_switcher context_switcher(TopMemoryContext);

    auto options = pg::table_options::current();
    pg::table_options::current().reset();
    auto [schema_name, simple_table_name] = split_table_name(table_name);
    if (table_exists(table_id) || table_exists(table_name)) {
        return;
    }

    std::string dataset_path;
    // Use provided dataset path or construct default path
    if (!options.dataset_path().empty()) {
        if (!pg::allow_custom_paths) {
            ereport(
                ERROR,
                (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
                 errmsg("Custom dataset_path is disabled"),
                 errhint("Set deeplake.allow_custom_paths=on or omit dataset_path and configure deeplake.root_path")));
        }
        // Explicit path provided via WITH clause
        dataset_path = options.dataset_path();
    } else {
        // Get session root path if set, otherwise fall back to default root.
        auto session_root = session_credentials::get_root_path();
        if (session_root.empty()) {
            session_root = pg::utils::get_deeplake_root_directory();
        }
        // Construct path: root_dir/db_name/schema_name/table_name
        // root_path is the global root; include the database name so each
        // database's datasets are stored under their own prefix.
        const auto db_name = get_current_database_name();
        dataset_path = session_root + "/" + db_name + "/" + schema_name + "/" + simple_table_name;
    }

    // Get credentials from current session
    auto creds = session_credentials::get_credentials();

    table_data td(table_id, table_name, CreateTupleDescCopy(tupdesc), dataset_path, creds);
    PinTupleDesc(td.get_tuple_descriptor());

    // Catalog-only mode: the dataset already exists on S3 (known from the catalog).
    // Skip all S3 operations — just register in pg_class (done by DDL) and our tables_ map.
    // Still write to pg_deeplake_tables so other code paths can discover the table locally.
    if (is_catalog_only_create()) {
        save_table_metadata(td);
        tables_.emplace(table_id, std::move(td));
        up_to_date_ = false;
        return;
    }

    bool ds_exists = false;
    try {
        auto creds_for_exists = creds;
        ds_exists = deeplake_api::exists(dataset_path, std::move(creds_for_exists)).get_future().get();
    } catch (const base::exception& e) {
        ereport(ERROR, (errcode(ERRCODE_INTERNAL_ERROR), errmsg("%s", e.what())));
    }

    try {
        if (ds_exists) {
            td.open_dataset(false);
            base::log_info(base::log_channel::generic,
                           "Table dataset exists, url: {}, opening, num_rows: {}",
                           td.get_dataset_path().url(),
                           td.num_rows());
            /// Validate columns
        } else {
            td.open_dataset(true);

            // Create columns based on TupleDesc
            for (auto i = 0; i < td.num_columns(); ++i) {
                const auto tupdesc_idx = td.get_tupdesc_index(i);
                Form_pg_attribute attr = TupleDescAttr(td.get_tuple_descriptor(), tupdesc_idx);
                const char* column_name = NameStr(attr->attname);
                // Resolve domain types to their base type
                Oid base_typeid = td.get_base_atttypid(i);
                // Map PostgreSQL types to DeepLake types
                switch (base_typeid) {
                case BOOLOID:
                    td.get_dataset()->add_column(column_name, nd::type::scalar(nd::dtype::boolean));
                    break;
                case INT2OID:
                    td.get_dataset()->add_column(column_name, nd::type::scalar(nd::dtype::int16));
                    break;
                case INT4OID:
                case DATEOID:
                    td.get_dataset()->add_column(column_name, nd::type::scalar(nd::dtype::int32));
                    break;
                case TIMEOID:
                case TIMESTAMPOID:
                case TIMESTAMPTZOID:
                case INT8OID:
                    td.get_dataset()->add_column(column_name, nd::type::scalar(nd::dtype::int64));
                    break;
                case FLOAT4OID:
                    td.get_dataset()->add_column(column_name, nd::type::scalar(nd::dtype::float32));
                    break;
                case NUMERICOID: {
                    const int32_t typmod = attr->atttypmod;
                    if (typmod >= 0) {
                        const int32_t precision = ((typmod - VARHDRSZ) >> 16) & 0xFFFF;
                        if (precision > 15) {
                            const int32_t scale = (typmod - VARHDRSZ) & 0xFFFF;
                            elog(WARNING,
                                 "Column '%s' has type NUMERIC(%d, %d), which may lose precision "
                                 "as it is stored as FLOAT64.",
                                 column_name,
                                 precision,
                                 scale);
                        }
                    }
                }
                case FLOAT8OID:
                    td.get_dataset()->add_column(column_name, nd::type::scalar(nd::dtype::float64));
                    break;
                case CHAROID:
                case BPCHAROID:
                case VARCHAROID: {
                    const int32_t typmod = attr->atttypmod;
                    if (typmod == VARHDRSZ + 1) {
                        td.get_dataset()->add_column(column_name,
                                                     deeplake_core::type::generic(nd::type::scalar(nd::dtype::int8)));
                    } else {
                        td.get_dataset()->add_column(column_name, deeplake_core::type::text(codecs::compression::null));
                    }
                    break;
                }
                case UUIDOID:
                case TEXTOID:
                    td.get_dataset()->add_column(column_name, deeplake_core::type::text(codecs::compression::null));
                    break;
                case JSONOID:
                case JSONBOID:
                    td.get_dataset()->add_column(column_name, deeplake_core::type::dict());
                    break;
                case BYTEAOID: {
                    // Check for special domain types over BYTEA
                    if (pg::utils::is_file_domain_type(attr->atttypid)) {
                        // FILE domain -> link of bytes
                        td.get_dataset()->add_column(
                            column_name,
                            deeplake_core::type::link(
                                deeplake_core::type::generic(nd::type::scalar(nd::dtype::byte))));
                        break;
                    }
                    if (pg::utils::is_image_domain_type(attr->atttypid)) {
                        // IMAGE domain -> image type
                        td.get_dataset()->add_column(
                            column_name,
                            deeplake_core::type::image(
                                nd::type::array(nd::dtype::byte, 3), codecs::compression::null));
                        break;
                    }
                    if (pg::utils::is_video_domain_type(attr->atttypid)) {
                        // VIDEO domain -> video type
                        td.get_dataset()->add_column(column_name,
                                                     deeplake_core::type::video(codecs::compression::mp4));
                        break;
                    }
                    td.get_dataset()->add_column(column_name,
                                                 deeplake_core::type::generic(nd::type::scalar(nd::dtype::byte)));
                    break;
                }
                case INT2ARRAYOID: {
                    int32_t ndims = (attr->attndims > 0) ? attr->attndims : 1;
                    if (ndims > 255) {
                        elog(ERROR,
                             "Column '%s' has unsupported type SMALLINT[] with %d dimensions (max 255)",
                             column_name,
                             ndims);
                    }
                    if (ndims == 1) {
                        td.get_dataset()->add_column(column_name, deeplake_core::type::embedding(0, nd::dtype::int16));
                    } else {
                        td.get_dataset()->add_column(
                            column_name, deeplake_core::type::generic(nd::type::array(nd::dtype::int16, ndims)));
                    }
                    break;
                }
                case INT4ARRAYOID: {
                    int32_t ndims = (attr->attndims > 0) ? attr->attndims : 1;
                    if (ndims > 255) {
                        elog(ERROR,
                             "Column '%s' has unsupported type INT[] with %d dimensions (max 255)",
                             column_name,
                             ndims);
                    }
                    if (ndims == 1) {
                        td.get_dataset()->add_column(column_name, deeplake_core::type::embedding(0, nd::dtype::int32));
                    } else {
                        td.get_dataset()->add_column(
                            column_name, deeplake_core::type::generic(nd::type::array(nd::dtype::int32, ndims)));
                    }
                    break;
                }
                case INT8ARRAYOID: {
                    int32_t ndims = (attr->attndims > 0) ? attr->attndims : 1;
                    if (ndims > 255) {
                        elog(ERROR,
                             "Column '%s' has unsupported type BIGINT[] with %d dimensions (max 255)",
                             column_name,
                             ndims);
                    }
                    if (ndims == 1) {
                        td.get_dataset()->add_column(column_name, deeplake_core::type::embedding(0, nd::dtype::int64));
                    } else {
                        td.get_dataset()->add_column(
                            column_name, deeplake_core::type::generic(nd::type::array(nd::dtype::int64, ndims)));
                    }
                    break;
                }
                case FLOAT4ARRAYOID: {
                    int32_t ndims = (attr->attndims > 0) ? attr->attndims : 1;
                    if (ndims > 255) {
                        elog(ERROR,
                             "Column '%s' has unsupported type REAL[] with %d dimensions (max 255)",
                             column_name,
                             ndims);
                    }
                    // Store FLOAT4[] as float32 (4 bytes)
                    if (ndims == 1) {
                        td.get_dataset()->add_column(column_name,
                                                     deeplake_core::type::embedding(0, nd::dtype::float32));
                    } else {
                        td.get_dataset()->add_column(
                            column_name, deeplake_core::type::generic(nd::type::array(nd::dtype::float32, ndims)));
                    }
                    break;
                }
                case FLOAT8ARRAYOID: {
                    int32_t ndims = (attr->attndims > 0) ? attr->attndims : 1;
                    if (ndims > 255) {
                        elog(ERROR,
                             "Column '%s' has unsupported type DOUBLE PRECISION[] with %d dimensions (max 255)",
                             column_name,
                             ndims);
                    }
                    if (ndims == 1) {
                        td.get_dataset()->add_column(column_name,
                                                     deeplake_core::type::embedding(0, nd::dtype::float64));
                    } else {
                        td.get_dataset()->add_column(
                            column_name, deeplake_core::type::generic(nd::type::array(nd::dtype::float64, ndims)));
                    }
                    break;
                }
                case BYTEAARRAYOID: {
                    if (attr->attndims > 1) {
                        elog(ERROR,
                             "Column '%s' has unsupported type BYTEA[] with %d dimensions",
                             column_name,
                             attr->attndims);
                    }
                    td.get_dataset()->add_column(
                        column_name, deeplake_core::type::generic(nd::type::array(nd::dtype::byte, attr->attndims)));
                    break;
                }
                case VARCHARARRAYOID:
                case TEXTARRAYOID: {
                    if (attr->attndims > 1) {
                        elog(ERROR,
                             "Column '%s' has unsupported type TEXT[] with %d dimensions",
                             column_name,
                             attr->attndims);
                    }
                    td.get_dataset()->add_column(
                        column_name, deeplake_core::type::generic(nd::type::array(nd::dtype::string, attr->attndims)));
                    break;
                }
                default: {
                    auto creds_for_delete = creds;
                    deeplake_api::delete_dataset(td.get_dataset_path(), std::move(creds_for_delete)).get_future().get();
                    const char* tname = format_type_with_typemod(attr->atttypid, attr->atttypmod);
                    elog(ERROR,
                         "Create Table: Column '%s' has unsupported type '%s' (OID %u, base OID %u)",
                         column_name,
                         tname,
                         attr->atttypid,
                         base_typeid);
                }
                }
            }

            base::log_info(base::log_channel::generic,
                           "Table dataset initialized with {} columns, url: {}",
                           td.num_columns(),
                           td.get_dataset_path().url());
            td.commit();
        }
        save_table_metadata(td);
    } catch (const base::exception& e) {
        ereport(ERROR, (errcode(ERRCODE_INTERNAL_ERROR), errmsg("%s", e.what())));
    }

    tables_.emplace(table_id, std::move(td));
    up_to_date_ = false;
}

void table_storage::drop_table(const std::string& table_name)
{
    // Load metadata BEFORE acquiring the DDL lock.
    // force_load_table_metadata() may trigger CREATE TABLE (via SPI) for tables
    // in the S3 catalog that don't exist in pg_class yet.  CREATE TABLE goes
    // through the table AM which also acquires the DDL lock — doing that while
    // we already hold it would self-deadlock (LWLocks are not recursive).
    if (!table_exists(table_name)) {
        force_load_table_metadata();
    }
    pg::table_ddl_lock_guard ddl_lock;
    if (table_exists(table_name)) {
        auto& table_data = get_table_data(table_name);
        auto creds = session_credentials::get_credentials();

        try {
            table_data.commit(); // Ensure all changes are committed before deletion
            table_version_tracker::drop_table(table_data.get_table_oid());
            deeplake_api::delete_dataset(table_data.get_dataset_path(), std::move(creds)).get_future().get();
        } catch (const storage::storage_key_not_found&) {
            // Dataset does not exist, silently continue with deletion
            elog(DEBUG1, "Dataset not found during table drop, continuing silently");
        } catch (const base::exception& e) {
            // Dataset deletion failed - log warning but continue with table cleanup
            // This handles cases where dataset was already deleted externally
            elog(WARNING, "Failed to delete dataset during table drop: %s", e.what());
        }
        erase_table(table_name);
    }
    /// FreeTupleDesc(table_data.tuple_descriptor_);
    erase_table_metadata(table_name);
}

void table_storage::insert_slot(Oid table_id, TupleTableSlot* slot)
{
    insert_slots(table_id, 1, &slot);
}

void table_storage::mark_subxact_change(Oid table_id)
{
    if (GetCurrentTransactionNestLevel() <= 1) {
        return;
    }

    auto& table_data = get_table_data(table_id);
    const SubTransactionId sub_id = GetCurrentSubTransactionId();
    auto& sub_snapshots = subxact_snapshots_[sub_id];
    if (!sub_snapshots.contains(table_id)) {
        sub_snapshots.emplace(table_id, table_data.capture_tx_snapshot());
    }
}

void table_storage::rollback_subxact(SubTransactionId sub_id)
{
    auto it = subxact_snapshots_.find(sub_id);
    if (it == subxact_snapshots_.end()) {
        return;
    }

    for (const auto& [table_id, snapshot] : it->second) {
        auto table_it = tables_.find(table_id);
        if (table_it != tables_.end()) {
            table_it->second.restore_tx_snapshot(snapshot);
        }
    }
    subxact_snapshots_.erase(it);
}

void table_storage::commit_subxact(SubTransactionId sub_id, SubTransactionId parent_sub_id)
{
    auto it = subxact_snapshots_.find(sub_id);
    if (it == subxact_snapshots_.end()) {
        return;
    }

    if (parent_sub_id != InvalidSubTransactionId) {
        auto& parent_snapshots = subxact_snapshots_[parent_sub_id];
        for (const auto& [table_id, snapshot] : it->second) {
            if (!parent_snapshots.contains(table_id)) {
                parent_snapshots.emplace(table_id, snapshot);
            }
        }
    }
    subxact_snapshots_.erase(it);
}

void table_storage::insert_slots(Oid table_id, int32_t nslots, TupleTableSlot** slots)
{
    mark_subxact_change(table_id);
    auto& table_data = get_table_data(table_id);
    table_data.add_insert_slots(nslots, slots);
}

bool table_storage::delete_tuple(Oid table_id, ItemPointer tid)
{
    mark_subxact_change(table_id);
    auto& table_data = get_table_data(table_id);
    try {
        const auto row_number = utils::tid_to_row_number(tid);

        // Verify row exists
        if (row_number >= table_data.num_rows()) {
            ereport(
                ERROR,
                (errcode(ERRCODE_INTERNAL_ERROR),
                 errmsg("Row number %zu out of range for table %s", row_number, table_data.get_table_name().c_str())));
            return false;
        }
        table_data.add_delete_row(row_number);
        return true;
    } catch (const base::exception& e) {
        ereport(ERROR, (errcode(ERRCODE_INTERNAL_ERROR), errmsg("Failed to delete tuple: %s", e.what())));
        return false;
    }
}

bool table_storage::update_tuple(Oid table_id, ItemPointer tid, HeapTuple new_tuple)
{
    mark_subxact_change(table_id);
    auto& table_data = get_table_data(table_id);
    TupleDesc tupdesc = table_data.get_tuple_descriptor();

    try {
        const auto row_number = utils::tid_to_row_number(tid);

        // Verify row exists
        if (row_number >= table_data.num_rows()) {
            ereport(
                ERROR,
                (errcode(ERRCODE_INTERNAL_ERROR),
                 errmsg("Row number %zu out of range for table %s", row_number, table_data.get_table_name().c_str())));
            return false;
        }

        // Extract values from HeapTuple
        std::vector<Datum> values(tupdesc->natts, 0);
        std::vector<uint8_t> nulls(tupdesc->natts, 0);
        heap_deform_tuple(new_tuple, tupdesc, values.data(), reinterpret_cast<bool*>(nulls.data()));

        icm::string_map<nd::array> row;
        ::convert_pg_to_nd(table_data, values, nulls, new_tuple->t_len, row);
        table_data.add_update_row(row_number, std::move(row));

        return true;
    } catch (const base::exception& e) {
        ereport(ERROR, (errcode(ERRCODE_INTERNAL_ERROR), errmsg("Failed to update tuple: %s", e.what())));
        return false;
    }
}

bool table_storage::fetch_tuple(Oid table_id, ItemPointer tid, TupleTableSlot* slot)
{
    auto& table_data = get_table_data(table_id);

    table_scan tscan(table_id, false, false);

    try {
        ExecClearTuple(slot);

        const auto row_number = utils::tid_to_row_number(tid);
        // Use num_total_rows() to include uncommitted rows in the current transaction.
        // This is necessary for AFTER triggers (like FK checks) that need to see
        // rows inserted earlier in the same transaction.
        if (row_number >= table_data.num_total_rows()) {
            return false;
        }
        Datum* values = slot->tts_values;
        bool* nulls = slot->tts_isnull;
        PG_TRY();
        {
            tscan.convert_nd_to_pg(row_number, values, nulls);
        }
        PG_CATCH();
        {
            PG_RE_THROW();
        }
        PG_END_TRY();

        ExecStoreVirtualTuple(slot);
        slot->tts_tid = *tid;
        return true;
    } catch (const base::exception& e) {
        ereport(ERROR, (errcode(ERRCODE_INTERNAL_ERROR), errmsg("Failed to fetch tuple: %s", e.what())));
    }
    return false;
}

void table_storage::clear()
{
    auto creds = session_credentials::get_credentials();
    // Clean up all datasets and metadata
    for (auto& [table_id, table_data] : tables_) {
        try {
            table_data.commit(); // Ensure all changes are committed before deletion
            auto creds_for_delete = creds;
            deeplake_api::delete_dataset(table_data.get_dataset_path(), std::move(creds_for_delete)).get_future().get();
            FreeTupleDesc(table_data.get_tuple_descriptor());
        } catch (const base::exception& e) {
            // Log error but continue cleaning up other tables
            ereport(
                WARNING,
                (errcode(ERRCODE_INTERNAL_ERROR),
                 errmsg("Failed to delete dataset for table %s: %s", table_data.get_table_name().c_str(), e.what())));
        }
    }
    table_version_tracker::clear_all_versions();
    tables_.clear();
    views_.clear();
    up_to_date_ = false;

    if (!pg::utils::check_table_exists("pg_deeplake_tables") || !pg::utils::check_table_exists("pg_deeplake_views")) {
        return;
    }
    // Clean up metadata table
    pg::utils::memory_context_switcher context_switcher;
    pg::utils::spi_connector connector;
    if (SPI_execute("DELETE FROM public.pg_deeplake_tables", false, 0) != SPI_OK_DELETE) {
        ereport(WARNING, (errcode(ERRCODE_INTERNAL_ERROR), errmsg("Failed to clear metadata table")));
    }
    if (SPI_execute("DELETE FROM public.pg_deeplake_views", false, 0) != SPI_OK_DELETE) {
        ereport(WARNING, (errcode(ERRCODE_INTERNAL_ERROR), errmsg("Failed to clear views metadata table")));
    }
}

table_data& table_storage::get_table_data(Oid table_id)
{
    if (!tables_.contains(table_id)) {
        // If not in memory, try loading from metadata
        force_load_table_metadata();
    }
    return tables_.at(table_id);
}

table_data& table_storage::get_table_data(const std::string& table_name)
{
    if (!table_exists(table_name)) {
        // If not in memory, try loading from metadata
        force_load_table_metadata();
    }
    for (auto& [_, table] : tables_) {
        if (table.get_table_name() == table_name) {
            return table;
        }
    }
    return tables_.at(InvalidOid);
}

void table_storage::add_view(Oid view_oid, const std::string& view_name, const std::string& view_str)
{
    if (view_exists(view_oid)) {
        return;
    }

    std::string insert_str =
        fmt::format("INSERT INTO public.pg_deeplake_views (view_name, query_string) VALUES ('{}', $${}$$) "
                    "ON CONFLICT (view_name) DO UPDATE SET query_string = EXCLUDED.query_string",
                    view_name,
                    view_str);

    pg::utils::spi_connector connector;
    if (SPI_execute(insert_str.c_str(), false, 0) != SPI_OK_INSERT) {
        throw pg::exception("Failed to save view metadata");
    }
    views_.emplace(view_oid, std::pair{view_name, view_str});
    up_to_date_ = false;
}

void table_storage::erase_view(const std::string& view_name)
{
    Oid view_oid = RelnameGetRelid(view_name.c_str());
    if (views_.erase(view_oid) == 0) {
        return;
    }
    up_to_date_ = false;
    pg::utils::memory_context_switcher context_switcher;
    if (!pg::utils::check_table_exists("pg_deeplake_views")) {
        return;
    }
    std::string delete_str = fmt::format("DELETE FROM public.pg_deeplake_views WHERE view_name = '{}'", view_name);

    pg::utils::spi_connector connector;
    if (SPI_execute(delete_str.c_str(), false, 0) != SPI_OK_DELETE) {
        ereport(ERROR, (errcode(ERRCODE_INTERNAL_ERROR), errmsg("Failed to erase view metadata")));
    }
}

} // namespace pg
