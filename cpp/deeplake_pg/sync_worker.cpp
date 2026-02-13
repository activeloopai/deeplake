#include "pg_deeplake.hpp"

#ifdef __cplusplus
extern "C" {
#endif

#include <postgres.h>

#include <access/xact.h>
#include <catalog/namespace.h>
#include <executor/spi.h>
#include <libpq-fe.h>
#include <miscadmin.h>
#include <nodes/makefuncs.h>
#include <pgstat.h>
#include <postmaster/bgworker.h>
#include <storage/ipc.h>
#include <storage/latch.h>
#include <storage/proc.h>
#include <tcop/utility.h>
#include <utils/builtins.h>
#include <utils/guc.h>
#include <utils/snapmgr.h>

#ifdef __cplusplus
}
#endif

#include "sync_worker.hpp"

#include "dl_catalog.hpp"
#include "table_storage.hpp"
#include "utils.hpp"

#include <async/promise.hpp>
#include <deeplake_api/catalog_table.hpp>
#include <icm/vector.hpp>

#include <algorithm>
#include <cstring>
#include <unordered_map>
#include <vector>

// GUC variables
int deeplake_sync_interval_ms = 1000;  // Default 1 second

// Forward declaration (defined in the anonymous namespace below)
namespace { bool execute_via_libpq(const char* dbname, const char* sql); }

// ---- pending_install_queue implementation ----

namespace pg {

pending_install_queue::queue_data* pending_install_queue::data_ = nullptr;

Size pending_install_queue::get_shmem_size()
{
    Size size = MAXALIGN(sizeof(queue_data));
    size = add_size(size, mul_size(MAX_PENDING, sizeof(entry)));
    return size;
}

void pending_install_queue::initialize()
{
    bool found = false;

    LWLockAcquire(AddinShmemInitLock, LW_EXCLUSIVE);

    data_ = static_cast<queue_data*>(ShmemInitStruct(
        "deeplake_pending_installs",
        get_shmem_size(),
        &found
    ));

    if (!found) {
        data_->lock = &(GetNamedLWLockTranche("deeplake_install_queue")->lock);
        data_->count = 0;
        memset(data_->entries, 0, MAX_PENDING * sizeof(entry));
    }

    LWLockRelease(AddinShmemInitLock);
}

bool pending_install_queue::enqueue(const char* dbname)
{
    if (data_ == nullptr || dbname == nullptr) {
        return false;
    }

    LWLockAcquire(data_->lock, LW_EXCLUSIVE);

    bool ok = false;
    if (data_->count < MAX_PENDING) {
        strlcpy(data_->entries[data_->count].db_name, dbname, NAMEDATALEN);
        data_->count++;
        ok = true;
    }

    LWLockRelease(data_->lock);
    return ok;
}

void pending_install_queue::drain_and_install()
{
    if (data_ == nullptr) {
        return;
    }

    // Copy entries under lock, then release before doing I/O
    std::vector<std::string> pending;

    LWLockAcquire(data_->lock, LW_EXCLUSIVE);
    for (int32_t i = 0; i < data_->count; i++) {
        pending.emplace_back(data_->entries[i].db_name);
    }
    data_->count = 0;
    LWLockRelease(data_->lock);

    // Install extension via libpq (outside any lock)
    for (const auto& db : pending) {
        if (execute_via_libpq(db.c_str(), "CREATE EXTENSION IF NOT EXISTS pg_deeplake")) {
            elog(LOG, "pg_deeplake: installed extension in database '%s' (async)", db.c_str());
        } else {
            elog(WARNING, "pg_deeplake: failed to install extension in database '%s' (async)", db.c_str());
        }
    }
}

} // namespace pg

namespace {

// Worker state - use sig_atomic_t for signal safety
volatile sig_atomic_t got_sigterm = false;
volatile sig_atomic_t got_sighup = false;

void deeplake_sync_worker_sigterm(SIGNAL_ARGS)
{
    int save_errno = errno;
    got_sigterm = true;
    SetLatch(MyLatch);
    errno = save_errno;
}

void deeplake_sync_worker_sighup(SIGNAL_ARGS)
{
    int save_errno = errno;
    got_sighup = true;
    SetLatch(MyLatch);
    errno = save_errno;
}

/**
 * Execute SQL via libpq in autocommit mode (needed for CREATE DATABASE which
 * cannot run inside a transaction block).
 *
 * Returns true on success. Treats SQLSTATE 42P04 ("duplicate_database") as success.
 */
bool execute_via_libpq(const char* dbname, const char* sql)
{
    // Build connection string using Unix socket
    const char* port = GetConfigOption("port", true, false);
    const char* socket_dir = GetConfigOption("unix_socket_directories", true, false);

    StringInfoData conninfo;
    initStringInfo(&conninfo);
    appendStringInfo(&conninfo, "dbname=%s", dbname);
    if (port) {
        appendStringInfo(&conninfo, " port=%s", port);
    }
    if (socket_dir) {
        // unix_socket_directories may be comma-separated; use the first one
        char* dir_copy = pstrdup(socket_dir);
        char* comma = strchr(dir_copy, ',');
        if (comma) *comma = '\0';
        // Trim leading/trailing whitespace
        char* dir = dir_copy;
        while (*dir == ' ') dir++;
        appendStringInfo(&conninfo, " host=%s", dir);
        pfree(dir_copy);
    }

    PGconn* conn = PQconnectdb(conninfo.data);
    pfree(conninfo.data);

    if (PQstatus(conn) != CONNECTION_OK) {
        elog(WARNING, "pg_deeplake sync: libpq connection failed: %s", PQerrorMessage(conn));
        PQfinish(conn);
        return false;
    }

    PGresult* res = PQexec(conn, sql);
    ExecStatusType status = PQresultStatus(res);
    bool ok = (status == PGRES_COMMAND_OK || status == PGRES_TUPLES_OK);

    if (!ok) {
        const char* sqlstate = PQresultErrorField(res, PG_DIAG_SQLSTATE);
        // 42P04 = duplicate_database - treat as success (idempotent)
        if (sqlstate && strcmp(sqlstate, "42P04") == 0) {
            ok = true;
        } else {
            elog(WARNING, "pg_deeplake sync: libpq exec failed: %s", PQerrorMessage(conn));
        }
    }

    PQclear(res);
    PQfinish(conn);
    return ok;
}

/**
 * Sync databases from the deeplake catalog to PostgreSQL.
 *
 * Creates missing databases and installs pg_deeplake extension in each.
 * Must be called OUTSIDE a transaction context since CREATE DATABASE
 * cannot run inside a transaction block.
 */
void deeplake_sync_databases_from_catalog(const std::string& root_path, icm::string_map<> creds)
{
    auto catalog_databases = pg::dl_catalog::load_databases(root_path, creds);

    for (const auto& db : catalog_databases) {
        // Skip system databases
        if (db.db_name == "postgres" || db.db_name == "template0" || db.db_name == "template1") {
            continue;
        }

        // Check if database already exists in pg_database via libpq
        // (we're outside a transaction, so use libpq to query postgres)
        StringInfoData check_sql;
        initStringInfo(&check_sql);
        appendStringInfo(&check_sql,
                         "SELECT 1 FROM pg_database WHERE datname = '%s'",
                         db.db_name.c_str());

        PGconn* conn = nullptr;
        {
            const char* port = GetConfigOption("port", true, false);
            const char* socket_dir = GetConfigOption("unix_socket_directories", true, false);

            StringInfoData conninfo;
            initStringInfo(&conninfo);
            appendStringInfo(&conninfo, "dbname=postgres");
            if (port) appendStringInfo(&conninfo, " port=%s", port);
            if (socket_dir) {
                char* dir_copy = pstrdup(socket_dir);
                char* comma = strchr(dir_copy, ',');
                if (comma) *comma = '\0';
                char* dir = dir_copy;
                while (*dir == ' ') dir++;
                appendStringInfo(&conninfo, " host=%s", dir);
                pfree(dir_copy);
            }
            conn = PQconnectdb(conninfo.data);
            pfree(conninfo.data);
        }

        if (PQstatus(conn) != CONNECTION_OK) {
            elog(WARNING, "pg_deeplake sync: cannot check database existence: %s", PQerrorMessage(conn));
            PQfinish(conn);
            pfree(check_sql.data);
            continue;
        }

        PGresult* res = PQexec(conn, check_sql.data);
        pfree(check_sql.data);
        bool exists = (PQresultStatus(res) == PGRES_TUPLES_OK && PQntuples(res) > 0);
        PQclear(res);
        PQfinish(conn);

        if (exists) {
            continue;
        }

        // Build CREATE DATABASE statement
        StringInfoData create_sql;
        initStringInfo(&create_sql);
        appendStringInfo(&create_sql, "CREATE DATABASE %s", quote_identifier(db.db_name.c_str()));
        if (!db.owner.empty()) {
            appendStringInfo(&create_sql, " OWNER %s", quote_identifier(db.owner.c_str()));
        }
        if (!db.encoding.empty()) {
            appendStringInfo(&create_sql, " ENCODING '%s'", db.encoding.c_str());
        }
        if (!db.lc_collate.empty()) {
            appendStringInfo(&create_sql, " LC_COLLATE '%s'", db.lc_collate.c_str());
        }
        if (!db.lc_ctype.empty()) {
            appendStringInfo(&create_sql, " LC_CTYPE '%s'", db.lc_ctype.c_str());
        }
        if (!db.template_db.empty()) {
            appendStringInfo(&create_sql, " TEMPLATE %s", quote_identifier(db.template_db.c_str()));
        }

        if (execute_via_libpq("postgres", create_sql.data)) {
            elog(LOG, "pg_deeplake sync: created database '%s'", db.db_name.c_str());

            // Install pg_deeplake extension in the new database
            if (execute_via_libpq(db.db_name.c_str(), "CREATE EXTENSION IF NOT EXISTS pg_deeplake")) {
                elog(LOG, "pg_deeplake sync: installed extension in database '%s'", db.db_name.c_str());
            } else {
                elog(WARNING, "pg_deeplake sync: failed to install extension in database '%s'", db.db_name.c_str());
            }
        } else {
            elog(WARNING, "pg_deeplake sync: failed to create database '%s'", db.db_name.c_str());
        }

        pfree(create_sql.data);
    }
}

/**
 * Sync schemas for a specific database from pre-loaded catalog data via libpq.
 * Creates missing schemas in the target database.
 */
void deeplake_sync_schemas_for_db(const std::string& db_name,
    const std::vector<pg::dl_catalog::schema_meta>& schemas)
{
    for (const auto& meta : schemas) {
        if (meta.state == "dropping") {
            continue;
        }

        // Skip system schemas
        if (meta.schema_name == "public" || meta.schema_name == "pg_catalog" ||
            meta.schema_name == "information_schema" ||
            meta.schema_name.substr(0, 3) == "pg_") {
            continue;
        }

        StringInfoData buf;
        initStringInfo(&buf);
        appendStringInfo(&buf, "CREATE SCHEMA IF NOT EXISTS %s",
                         quote_identifier(meta.schema_name.c_str()));

        if (execute_via_libpq(db_name.c_str(), buf.data)) {
            elog(LOG, "pg_deeplake sync: created schema '%s' in database '%s'",
                 meta.schema_name.c_str(), db_name.c_str());
        }

        pfree(buf.data);
    }
}

/**
 * Sync tables for a specific database from pre-loaded catalog data via libpq.
 * Creates missing tables in the target database.
 */
/**
 * Parse comma-separated column names string into a vector.
 * The column_names string uses trailing comma format: "col1,col2,"
 */
std::vector<std::string> parse_column_names(const std::string& column_names)
{
    std::vector<std::string> result;
    std::string current;
    for (char c : column_names) {
        if (c == ',') {
            if (!current.empty()) {
                result.push_back(current);
                current.clear();
            }
        } else {
            current += c;
        }
    }
    if (!current.empty()) {
        result.push_back(current);
    }
    return result;
}

void deeplake_sync_tables_for_db(const std::string& db_name,
    const std::vector<pg::dl_catalog::table_meta>& tables,
    const std::vector<pg::dl_catalog::column_meta>& columns,
    const std::vector<pg::dl_catalog::index_meta>& indexes)
{
    for (const auto& meta : tables) {
        if (meta.state == "dropping") {
            continue;
        }

        const std::string qualified_name = meta.schema_name + "." + meta.table_name;

        // Gather columns for this table, sorted by position
        std::vector<pg::dl_catalog::column_meta> table_columns;
        for (const auto& col : columns) {
            if (col.table_id == meta.table_id) {
                table_columns.push_back(col);
            }
        }
        std::sort(table_columns.begin(), table_columns.end(),
                  [](const auto& a, const auto& b) { return a.position < b.position; });

        if (table_columns.empty()) {
            elog(DEBUG1, "pg_deeplake sync: no columns for %s in db %s, skipping",
                 qualified_name.c_str(), db_name.c_str());
            continue;
        }

        // Find indexes for this table
        std::vector<pg::dl_catalog::index_meta> table_indexes;
        for (const auto& idx : indexes) {
            if (idx.table_id == meta.table_id) {
                table_indexes.push_back(idx);
            }
        }

        // Determine which columns are part of a primary key (inverted_index on non-nullable columns)
        // The primary key columns are stored as comma-separated names in column_names
        std::vector<std::string> pk_columns;
        for (const auto& idx : table_indexes) {
            if (idx.index_type == "inverted_index") {
                pk_columns = parse_column_names(idx.column_names);
                break;
            }
        }

        const char* qschema = quote_identifier(meta.schema_name.c_str());
        const char* qtable = quote_identifier(meta.table_name.c_str());

        // Combine schema + table creation into a single SQL statement
        StringInfoData buf;
        initStringInfo(&buf);
        appendStringInfo(&buf, "CREATE SCHEMA IF NOT EXISTS %s; ", qschema);
        appendStringInfo(&buf, "CREATE TABLE IF NOT EXISTS %s.%s (", qschema, qtable);

        bool first = true;
        for (const auto& col : table_columns) {
            if (!first) {
                appendStringInfoString(&buf, ", ");
            }
            first = false;
            appendStringInfo(&buf, "%s %s", quote_identifier(col.column_name.c_str()), col.pg_type.c_str());
        }

        // Add PRIMARY KEY table constraint if we have PK columns
        if (!pk_columns.empty()) {
            appendStringInfoString(&buf, ", PRIMARY KEY (");
            for (size_t i = 0; i < pk_columns.size(); ++i) {
                if (i > 0) {
                    appendStringInfoString(&buf, ", ");
                }
                appendStringInfoString(&buf, quote_identifier(pk_columns[i].c_str()));
            }
            appendStringInfoChar(&buf, ')');
        }

        appendStringInfo(&buf, ") USING deeplake");

        if (execute_via_libpq(db_name.c_str(), buf.data)) {
            elog(LOG, "pg_deeplake sync: created table %s in database %s",
                 qualified_name.c_str(), db_name.c_str());
        }

        pfree(buf.data);
    }
}

/**
 * Sync all databases: check per-db versions in parallel, load changed ones,
 * create missing tables via libpq.
 *
 * Called OUTSIDE transaction context.
 */
void sync_all_databases(
    const std::string& root_path,
    icm::string_map<> creds,
    std::unordered_map<std::string, int64_t>& last_db_versions)
{
    // Step 1: Sync databases (create missing ones, install extension)
    deeplake_sync_databases_from_catalog(root_path, creds);

    // Step 2: Get list of all databases from the shared catalog
    auto databases = pg::dl_catalog::load_databases(root_path, creds);

    // Always include "postgres" which may not be in the databases catalog
    bool has_postgres = false;
    for (const auto& db : databases) {
        if (db.db_name == "postgres") { has_postgres = true; break; }
    }
    if (!has_postgres) {
        pg::dl_catalog::database_meta pg_meta;
        pg_meta.db_name = "postgres";
        pg_meta.state = "ready";
        databases.push_back(std::move(pg_meta));
    }

    // Step 3: Open per-db meta tables and check versions in parallel
    std::vector<std::string> db_names;
    std::vector<std::shared_ptr<deeplake_api::catalog_table>> meta_handles;

    for (const auto& db : databases) {
        if (db.db_name == "template0" || db.db_name == "template1") {
            continue;
        }
        try {
            auto handle = pg::dl_catalog::open_db_meta_table(root_path, db.db_name, creds);
            if (handle) {
                db_names.push_back(db.db_name);
                meta_handles.push_back(std::move(handle));
            }
        } catch (...) {
            // Per-db catalog may not exist yet — skip silently
            elog(DEBUG1, "pg_deeplake sync: no per-db catalog for '%s', skipping", db.db_name.c_str());
        }
    }

    if (db_names.empty()) {
        return;
    }

    // Fire all version() promises in parallel (1 round-trip wall-clock)
    icm::vector<async::promise<uint64_t>> version_promises;
    version_promises.reserve(db_names.size());
    for (auto& handle : meta_handles) {
        version_promises.push_back(handle->version());
    }
    auto versions = async::combine(std::move(version_promises)).get_future().get();

    // Step 4: Identify databases whose version changed since last sync
    std::vector<std::string> changed_dbs;
    for (size_t i = 0; i < db_names.size(); ++i) {
        int64_t ver = static_cast<int64_t>(versions[i]);
        auto it = last_db_versions.find(db_names[i]);
        if (it == last_db_versions.end() || it->second != ver) {
            changed_dbs.push_back(db_names[i]);
            last_db_versions[db_names[i]] = ver;
        }
    }

    if (changed_dbs.empty()) {
        return;
    }

    // Step 5: For each changed database, load schemas first, then tables+columns and sync
    for (const auto& db_name : changed_dbs) {
        try {
            // Sync schemas before tables so CREATE TABLE can find the target schema
            auto schemas = pg::dl_catalog::load_schemas(root_path, db_name, creds);
            if (!schemas.empty()) {
                deeplake_sync_schemas_for_db(db_name, schemas);
            }

            auto [tables, columns] = pg::dl_catalog::load_tables_and_columns(root_path, db_name, creds);
            auto indexes = pg::dl_catalog::load_indexes(root_path, db_name, creds);
            deeplake_sync_tables_for_db(db_name, tables, columns, indexes);
            elog(LOG, "pg_deeplake sync: synced %zu schemas, %zu tables, %zu indexes for database '%s'",
                 schemas.size(), tables.size(), indexes.size(), db_name.c_str());
        } catch (const std::exception& e) {
            elog(WARNING, "pg_deeplake sync: failed to sync database '%s': %s", db_name.c_str(), e.what());
        } catch (...) {
            elog(WARNING, "pg_deeplake sync: failed to sync database '%s': unknown error", db_name.c_str());
        }
    }
}

} // anonymous namespace

extern "C" {

PGDLLEXPORT void deeplake_sync_worker_main(Datum main_arg)
{
    // Set up signal handlers
    pqsignal(SIGTERM, deeplake_sync_worker_sigterm);
    pqsignal(SIGHUP, deeplake_sync_worker_sighup);

    // Unblock signals
    BackgroundWorkerUnblockSignals();

    // Connect to the default database
    BackgroundWorkerInitializeConnection("postgres", NULL, 0);

    elog(LOG, "pg_deeplake sync worker started");

    int64_t last_catalog_version = 0;
    std::string last_root_path;  // Track root_path to detect changes
    std::unordered_map<std::string, int64_t> last_db_versions;


    while (!got_sigterm) {
        // Process pending interrupts (including ProcSignalBarrier from DROP DATABASE)
        CHECK_FOR_INTERRUPTS();

        // Handle SIGHUP - reload configuration
        if (got_sighup) {
            got_sighup = false;
            ProcessConfigFile(PGC_SIGHUP);
        }

        // Variables to carry state across transaction boundaries
        // (declared before goto target to avoid crossing initialization)
        std::string sync_root_path;
        icm::string_map<> sync_creds;
        bool need_sync = false;

        // Skip if stateless mode is disabled
        if (!pg::stateless_enabled) {
            goto wait_for_latch;
        }

        // Start a transaction for our work
        SetCurrentStatementStartTimestamp();
        StartTransactionCommand();
        PushActiveSnapshot(GetTransactionSnapshot());

        PG_TRY();
        {
            // Initialize DeepLake (loads table metadata, etc.)
            pg::init_deeplake();

            auto root_path = pg::session_credentials::get_root_path();
            if (root_path.empty()) {
                root_path = pg::utils::get_deeplake_root_directory();
            }

            if (!root_path.empty()) {
                auto creds = pg::session_credentials::get_credentials();

                // When root_path changes after initial setup, force a full reload
                if (root_path != last_root_path) {
                    if (!last_root_path.empty()) {
                        pg::table_storage::instance().reset_and_load_table_metadata();
                        last_catalog_version = 0;
                        last_db_versions.clear();
                    }
                    last_root_path = root_path;
                }

                // Fast global version check (single HEAD request via cached meta table)
                int64_t current_version = pg::dl_catalog::get_catalog_version(root_path, creds);

                if (current_version != last_catalog_version) {
                    // Save state for sync (which happens outside transaction)
                    sync_root_path = root_path;
                    sync_creds = creds;
                    need_sync = true;
                    last_catalog_version = current_version;
                }
            }
        }
        PG_CATCH();
        {
            // Log error but don't crash - continue polling
            EmitErrorReport();
            FlushErrorState();
        }
        PG_END_TRY();

        PopActiveSnapshot();
        CommitTransactionCommand();

        // All sync happens OUTSIDE transaction context via libpq
        if (need_sync && !sync_root_path.empty()) {
            try {
                sync_all_databases(sync_root_path, sync_creds, last_db_versions);
                elog(DEBUG1, "pg_deeplake sync: completed (global version %ld)", last_catalog_version);
            } catch (const std::exception& e) {
                elog(WARNING, "pg_deeplake sync: sync failed: %s", e.what());
            } catch (...) {
                elog(WARNING, "pg_deeplake sync: sync failed: unknown error");
            }
        }

        pgstat_report_stat(true);

        // Drain any databases queued for async extension install
        pg::pending_install_queue::drain_and_install();

    wait_for_latch:
        // Wait for latch or timeout
        (void)WaitLatch(MyLatch,
                        WL_LATCH_SET | WL_TIMEOUT | WL_EXIT_ON_PM_DEATH,
                        deeplake_sync_interval_ms,
                        PG_WAIT_EXTENSION);
        ResetLatch(MyLatch);
    }

    elog(LOG, "pg_deeplake sync worker shutting down");
    proc_exit(0);
}

} // extern "C"
