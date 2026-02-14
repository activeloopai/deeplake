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

#include "dl_wal.hpp"
#include "table_storage.hpp"
#include "utils.hpp"

#include <async/promise.hpp>
#include <deeplake_api/catalog_table.hpp>
#include <icm/vector.hpp>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <unistd.h>
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

std::string wal_checkpoint_file_path()
{
    if (DataDir == nullptr) {
        return "/tmp/pg_deeplake_wal_checkpoints.tsv";
    }
    return std::string(DataDir) + "/pg_deeplake_wal_checkpoints.tsv";
}

std::string checkpoint_key(const std::string& root_path, const std::string& db_name)
{
    return root_path + "\t" + db_name;
}

void load_wal_checkpoints(std::unordered_map<std::string, int64_t>& checkpoints)
{
    checkpoints.clear();
    std::ifstream in(wal_checkpoint_file_path());
    if (!in.is_open()) {
        return;
    }

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        std::istringstream ss(line);
        std::string root_path;
        std::string db_name;
        std::string seq_str;
        if (!std::getline(ss, root_path, '\t') ||
            !std::getline(ss, db_name, '\t') ||
            !std::getline(ss, seq_str)) {
            continue;
        }
        try {
            checkpoints[checkpoint_key(root_path, db_name)] = std::stoll(seq_str);
        } catch (...) {
        }
    }
}

void persist_wal_checkpoints(const std::unordered_map<std::string, int64_t>& checkpoints)
{
    const std::string path = wal_checkpoint_file_path();
    const std::string tmp_path = path + ".tmp";

    std::ofstream out(tmp_path, std::ios::trunc);
    if (!out.is_open()) {
        elog(WARNING, "pg_deeplake sync: failed to open WAL checkpoint tmp file: %s", tmp_path.c_str());
        return;
    }

    for (const auto& kv : checkpoints) {
        const size_t sep = kv.first.find('\t');
        if (sep == std::string::npos) {
            continue;
        }
        out << kv.first.substr(0, sep) << '\t'
            << kv.first.substr(sep + 1) << '\t'
            << kv.second << '\n';
    }
    out.close();

    if (std::rename(tmp_path.c_str(), path.c_str()) != 0) {
        elog(WARNING, "pg_deeplake sync: failed to persist WAL checkpoints to %s", path.c_str());
    }
}

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
    auto catalog_databases = pg::dl_wal::load_databases(root_path, creds);

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

bool is_harmless_replay_error(const char* sqlstate)
{
    if (sqlstate == nullptr) {
        return false;
    }
    return strcmp(sqlstate, "42P07") == 0 ||  // duplicate_table
           strcmp(sqlstate, "42P06") == 0 ||  // duplicate_schema
           strcmp(sqlstate, "42701") == 0 ||  // duplicate_column
           strcmp(sqlstate, "42710") == 0 ||  // duplicate_object
           strcmp(sqlstate, "42704") == 0 ||  // undefined_object
           strcmp(sqlstate, "42P01") == 0 ||  // undefined_table
           strcmp(sqlstate, "3F000") == 0;    // invalid_schema_name
}

void deeplake_replay_ddl_log_for_db(const std::string& db_name, const std::string& root_path,
                                    icm::string_map<> creds, int64_t& last_seq)
{
    auto entries = pg::dl_wal::load_ddl_log(root_path, db_name, creds, last_seq);
    for (const auto& entry : entries) {
        if (entry.seq > last_seq) {
            last_seq = entry.seq;
        }
        if (entry.origin_instance_id == pg::dl_wal::local_instance_id()) {
            continue;
        }

        StringInfoData sql;
        initStringInfo(&sql);
        appendStringInfo(&sql, "SET application_name = 'pg_deeplake_sync'; ");
        if (!entry.search_path.empty()) {
            appendStringInfo(&sql,
                             "SELECT pg_catalog.set_config('search_path', %s, false); ",
                             quote_literal_cstr(entry.search_path.c_str()));
        }
        appendStringInfoString(&sql, entry.ddl_sql.c_str());

        const char* port = GetConfigOption("port", true, false);
        const char* socket_dir = GetConfigOption("unix_socket_directories", true, false);

        StringInfoData conninfo;
        initStringInfo(&conninfo);
        appendStringInfo(&conninfo, "dbname=%s", db_name.c_str());
        if (port != nullptr) {
            appendStringInfo(&conninfo, " port=%s", port);
        }
        if (socket_dir != nullptr) {
            char* dir_copy = pstrdup(socket_dir);
            char* comma = strchr(dir_copy, ',');
            if (comma != nullptr) {
                *comma = '\0';
            }
            char* dir = dir_copy;
            while (*dir == ' ') {
                dir++;
            }
            appendStringInfo(&conninfo, " host=%s", dir);
            pfree(dir_copy);
        }

        PGconn* conn = PQconnectdb(conninfo.data);
        pfree(conninfo.data);
        if (PQstatus(conn) != CONNECTION_OK) {
            elog(WARNING, "pg_deeplake sync: libpq connect failed for '%s': %s", db_name.c_str(), PQerrorMessage(conn));
            PQfinish(conn);
            pfree(sql.data);
            continue;
        }

        PGresult* res = PQexec(conn, sql.data);
        const ExecStatusType status = PQresultStatus(res);
        bool ok = status == PGRES_COMMAND_OK || status == PGRES_TUPLES_OK;
        if (!ok) {
            const char* sqlstate = PQresultErrorField(res, PG_DIAG_SQLSTATE);
            if (!is_harmless_replay_error(sqlstate)) {
                elog(WARNING,
                     "pg_deeplake sync: DDL WAL replay failed in '%s' [%s]: %s (SQL: %.200s)",
                     db_name.c_str(),
                     entry.command_tag.c_str(),
                     PQerrorMessage(conn),
                     entry.ddl_sql.c_str());
            } else {
                ok = true;
            }
        }
        if (ok) {
            elog(LOG, "pg_deeplake sync: replayed %s in '%s'", entry.command_tag.c_str(), db_name.c_str());
        }
        PQclear(res);
        PQfinish(conn);
        pfree(sql.data);
    }
}

/**
 * Sync all databases: check per-db versions in parallel, replay new DDL WAL entries.
 *
 * Called OUTSIDE transaction context.
 */
void sync_all_databases(
    const std::string& root_path,
    icm::string_map<> creds,
    std::unordered_map<std::string, int64_t>& last_db_seqs)
{
    // Step 1: Sync databases (create missing ones, install extension)
    deeplake_sync_databases_from_catalog(root_path, creds);

    // Step 2: Get list of all databases from the shared catalog
    auto databases = pg::dl_wal::load_databases(root_path, creds);

    // Always include "postgres" which may not be in the databases catalog
    bool has_postgres = false;
    for (const auto& db : databases) {
        if (db.db_name == "postgres") { has_postgres = true; break; }
    }
    if (!has_postgres) {
        pg::dl_wal::database_meta pg_meta;
        pg_meta.db_name = "postgres";
        pg_meta.state = "ready";
        databases.push_back(std::move(pg_meta));
    }

    // Step 3: For each database, replay DDL WAL entries
    // (cheap if no new entries due to after_seq filtering)
    bool checkpoints_updated = false;
    for (const auto& db : databases) {
        if (db.db_name == "template0" || db.db_name == "template1") {
            continue;
        }
        try {
            const std::string key = checkpoint_key(root_path, db.db_name);
            int64_t& last_seq = last_db_seqs[key];
            const int64_t prev_seq = last_seq;
            deeplake_replay_ddl_log_for_db(db.db_name, root_path, creds, last_seq);
            if (last_seq != prev_seq) {
                checkpoints_updated = true;
            }
            elog(LOG, "pg_deeplake sync: replayed DDL WAL for '%s' (last_seq=%ld)", db.db_name.c_str(), last_seq);
        } catch (const std::exception& e) {
            elog(WARNING, "pg_deeplake sync: failed to sync database '%s': %s", db.db_name.c_str(), e.what());
        } catch (...) {
            elog(WARNING, "pg_deeplake sync: failed to sync database '%s': unknown error", db.db_name.c_str());
        }
    }

    if (checkpoints_updated) {
        persist_wal_checkpoints(last_db_seqs);
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
    std::unordered_map<std::string, int64_t> last_db_seqs;
    load_wal_checkpoints(last_db_seqs);

    while (!got_sigterm) {
        // Process pending interrupts (including ProcSignalBarrier from DROP DATABASE)
        CHECK_FOR_INTERRUPTS();

        // Handle SIGHUP - reload configuration
        if (got_sighup) {
            got_sighup = false;
            ProcessConfigFile(PGC_SIGHUP);
        }

        // Always drain pending extension installs first so CREATE DATABASE
        // async installs are not starved behind expensive sync work.
        pg::pending_install_queue::drain_and_install();

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
                        last_db_seqs.clear();
                    }
                    last_root_path = root_path;
                }

                // Fast global version check via databases catalog_table
                int64_t current_version = pg::dl_wal::get_databases_version(root_path, creds);

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
                sync_all_databases(sync_root_path, sync_creds, last_db_seqs);
                elog(DEBUG1, "pg_deeplake sync: completed (global version %ld)", last_catalog_version);
            } catch (const std::exception& e) {
                elog(WARNING, "pg_deeplake sync: sync failed: %s", e.what());
            } catch (...) {
                elog(WARNING, "pg_deeplake sync: sync failed: unknown error");
            }
        }

        pgstat_report_stat(true);

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
