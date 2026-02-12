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

#include <algorithm>
#include <cstring>
#include <vector>

// GUC variables
int deeplake_sync_interval_ms = 2000;  // Default 2 seconds

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
 * Sync tables from the deeplake catalog to PostgreSQL.
 *
 * This function checks the catalog for tables that exist in the deeplake
 * catalog but not in PostgreSQL, and creates them.
 */
void deeplake_sync_tables_from_catalog(const std::string& root_path, icm::string_map<> creds)
{
    // Load tables and columns in parallel for better performance
    auto [catalog_tables, catalog_columns] = pg::dl_catalog::load_tables_and_columns(root_path, creds);

    for (const auto& meta : catalog_tables) {
        // Skip tables marked as dropping
        if (meta.state == "dropping") {
            continue;
        }

        const std::string qualified_name = meta.schema_name + "." + meta.table_name;

        // Check if table exists in PostgreSQL
        auto* rel = makeRangeVar(pstrdup(meta.schema_name.c_str()), pstrdup(meta.table_name.c_str()), -1);
        Oid relid = RangeVarGetRelid(rel, NoLock, true);

        if (!OidIsValid(relid)) {
            // Gather columns for this table, sorted by position
            std::vector<pg::dl_catalog::column_meta> table_columns;
            for (const auto& col : catalog_columns) {
                if (col.table_id == meta.table_id) {
                    table_columns.push_back(col);
                }
            }
            std::sort(table_columns.begin(), table_columns.end(),
                      [](const auto& a, const auto& b) { return a.position < b.position; });

            if (table_columns.empty()) {
                elog(DEBUG1, "pg_deeplake sync: no columns found for table %s, skipping", qualified_name.c_str());
                continue;
            }

            const char* qschema = quote_identifier(meta.schema_name.c_str());
            const char* qtable = quote_identifier(meta.table_name.c_str());

            // Build CREATE TABLE IF NOT EXISTS statement
            StringInfoData buf;
            initStringInfo(&buf);
            appendStringInfo(&buf, "CREATE TABLE IF NOT EXISTS %s.%s (", qschema, qtable);

            bool first = true;
            for (const auto& col : table_columns) {
                if (!first) {
                    appendStringInfoString(&buf, ", ");
                }
                first = false;
                appendStringInfo(&buf, "%s %s", quote_identifier(col.column_name.c_str()), col.pg_type.c_str());
            }
            appendStringInfo(&buf, ") USING deeplake");

            // Wrap in subtransaction so that if another backend concurrently
            // creates the same table (race on composite type), the error is
            // caught and we continue instead of aborting the sync cycle.
            MemoryContext saved_context = CurrentMemoryContext;
            ResourceOwner saved_owner = CurrentResourceOwner;

            BeginInternalSubTransaction(NULL);
            PG_TRY();
            {
                pg::utils::spi_connector connector;

                // Create schema if needed
                StringInfoData schema_buf;
                initStringInfo(&schema_buf);
                appendStringInfo(&schema_buf, "CREATE SCHEMA IF NOT EXISTS %s", qschema);
                SPI_execute(schema_buf.data, false, 0);
                pfree(schema_buf.data);

                if (SPI_execute(buf.data, false, 0) == SPI_OK_UTILITY) {
                    elog(LOG, "pg_deeplake sync: successfully created table %s", qualified_name.c_str());
                }

                ReleaseCurrentSubTransaction();
            }
            PG_CATCH();
            {
                // Another backend created this table concurrently — not an error.
                MemoryContextSwitchTo(saved_context);
                CurrentResourceOwner = saved_owner;
                RollbackAndReleaseCurrentSubTransaction();
                FlushErrorState();
                elog(DEBUG1, "pg_deeplake sync: concurrent creation of %s, skipping", qualified_name.c_str());
            }
            PG_END_TRY();

            pfree(buf.data);
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
                    }
                    last_root_path = root_path;
                }

                // Use existing catalog version API to check for changes (now fast with cache)
                int64_t current_version = pg::dl_catalog::get_catalog_version(root_path, creds);

                if (current_version != last_catalog_version) {
                    // Save state for database sync (which happens outside transaction)
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

        // Sync databases via libpq OUTSIDE transaction context
        // (CREATE DATABASE cannot run inside a transaction block)
        if (need_sync && !sync_root_path.empty()) {
            try {
                deeplake_sync_databases_from_catalog(sync_root_path, sync_creds);
            } catch (const std::exception& e) {
                elog(WARNING, "pg_deeplake sync: database sync failed: %s", e.what());
            } catch (...) {
                elog(WARNING, "pg_deeplake sync: database sync failed: unknown error");
            }

            // Re-enter transaction for table sync
            SetCurrentStatementStartTimestamp();
            StartTransactionCommand();
            PushActiveSnapshot(GetTransactionSnapshot());

            PG_TRY();
            {
                deeplake_sync_tables_from_catalog(sync_root_path, sync_creds);
                elog(DEBUG1, "pg_deeplake sync: synced (catalog version %ld)", last_catalog_version);
            }
            PG_CATCH();
            {
                EmitErrorReport();
                FlushErrorState();
            }
            PG_END_TRY();

            PopActiveSnapshot();
            CommitTransactionCommand();
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
