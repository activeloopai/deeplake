#include "pg_deeplake.hpp"

#ifdef __cplusplus
extern "C" {
#endif

#include <postgres.h>

#include <access/xact.h>
#include <catalog/namespace.h>
#include <executor/spi.h>
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
#include <vector>

// GUC variables
int deeplake_sync_interval_ms = 2000;  // Default 2 seconds
bool deeplake_sync_enabled = true;

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
 * Sync tables from the deeplake catalog to PostgreSQL.
 *
 * This function checks the catalog for tables that exist in the deeplake
 * catalog but not in PostgreSQL, and creates them.
 */
void deeplake_sync_tables_from_catalog(const std::string& root_path, icm::string_map<> creds)
{
    auto catalog_tables = pg::dl_catalog::load_tables(root_path, creds);
    auto catalog_columns = pg::dl_catalog::load_columns(root_path, creds);

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
            // Table doesn't exist locally - create it
            elog(LOG, "pg_deeplake sync: creating table %s from catalog", qualified_name.c_str());

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

            StringInfoData buf;
            initStringInfo(&buf);

            // Create schema if needed
            appendStringInfo(&buf, "CREATE SCHEMA IF NOT EXISTS %s", qschema);

            pg::utils::spi_connector connector;
            if (SPI_execute(buf.data, false, 0) != SPI_OK_UTILITY) {
                elog(WARNING, "pg_deeplake sync: failed to create schema %s", meta.schema_name.c_str());
                pfree(buf.data);
                continue;
            }

            // Build CREATE TABLE statement directly from catalog metadata
            // This avoids calling the SQL function create_deeplake_table which may not exist
            // in the postgres database (extension might not be installed there)
            resetStringInfo(&buf);
            const char* qtable = quote_identifier(meta.table_name.c_str());
            appendStringInfo(&buf, "CREATE TABLE %s.%s (", qschema, qtable);

            bool first = true;
            for (const auto& col : table_columns) {
                if (!first) {
                    appendStringInfoString(&buf, ", ");
                }
                first = false;
                appendStringInfo(&buf, "%s %s", quote_identifier(col.column_name.c_str()), col.pg_type.c_str());
            }

            // Table path is now derived from deeplake.root_path GUC set at database level
            // Path: {root_path}/{schema}/{table_name}
            appendStringInfo(&buf, ") USING deeplake");

            if (SPI_execute(buf.data, false, 0) != SPI_OK_UTILITY) {
                // Don't log as warning - the dataset might not be available yet
                // The sync worker will retry on the next cycle
                elog(DEBUG1, "pg_deeplake sync: table %s not ready yet, will retry", qualified_name.c_str());
            } else {
                elog(LOG, "pg_deeplake sync: successfully created table %s", qualified_name.c_str());
            }

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

    while (!got_sigterm) {
        // Handle SIGHUP - reload configuration
        if (got_sighup) {
            got_sighup = false;
            ProcessConfigFile(PGC_SIGHUP);
        }

        // Skip if sync is disabled
        if (!deeplake_sync_enabled) {
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

            // Only sync from catalog when stateless mode is enabled
            if (pg::stateless_enabled) {
                auto root_path = pg::session_credentials::get_root_path();
                if (root_path.empty()) {
                    root_path = pg::utils::get_deeplake_root_directory();
                }

                if (!root_path.empty()) {
                    auto creds = pg::session_credentials::get_credentials();

                    // Ensure catalog exists
                    pg::dl_catalog::ensure_catalog(root_path, creds);

                    // Use existing catalog version API to check for changes
                    int64_t current_version = pg::dl_catalog::get_catalog_version(root_path, creds);

                    if (current_version != last_catalog_version) {
                        // Version changed - sync tables from catalog
                        deeplake_sync_tables_from_catalog(root_path, creds);
                        last_catalog_version = current_version;
                        elog(LOG, "pg_deeplake sync: synced tables (catalog version %ld)", current_version);
                    }
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
