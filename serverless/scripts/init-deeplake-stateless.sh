#!/bin/bash
# init-deeplake-stateless.sh
# Mounted as /docker-entrypoint-initdb.d/3-stateless-init.sh
# Runs once during container first boot (initdb phase).
#
# Configures pg_deeplake for stateless S3 mode via ALTER SYSTEM so the
# settings persist across restarts without needing SET on every connection.
#
# NOTE: Credentials (deeplake.creds) are NOT set here. They are written to
# /dev/shm/deeplake-creds.conf by deeplake-entrypoint.sh on every container
# start, and included via include_if_exists in postgresql-overrides.conf.
# This ensures credentials never touch persistent storage.

set -euo pipefail

echo "=== pg_deeplake stateless init ==="

# Non-sensitive settings only — persist across restarts via postgresql.auto.conf
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    ALTER SYSTEM SET deeplake.stateless_enabled = true;
    ALTER SYSTEM SET deeplake.root_path = '${DEEPLAKE_ROOT_PATH}';
    ALTER SYSTEM SET deeplake.sync_interval_ms = ${DEEPLAKE_SYNC_INTERVAL_MS:-1000};
    ALTER SYSTEM SET pg_deeplake.memory_limit_mb = ${PG_DEEPLAKE_MEMORY_LIMIT_MB:-0};
EOSQL

# Append performance overrides to postgresql.conf
# (ALTER SYSTEM can't set some params before the postmaster reads them)
if [ -f /etc/postgresql-overrides.conf ]; then
    echo "" >> "$PGDATA/postgresql.conf"
    echo "# --- serverless overrides ---" >> "$PGDATA/postgresql.conf"
    cat /etc/postgresql-overrides.conf >> "$PGDATA/postgresql.conf"
    echo "Appended postgresql-overrides.conf to postgresql.conf"
fi

# No pg_reload_conf() here — ALTER SYSTEM settings in postgresql.auto.conf
# are picked up automatically when PostgreSQL restarts after the initdb phase.
# Reloading during initdb would start the sync worker prematurely, causing
# a ~60s shutdown delay while it finishes S3 I/O.

echo "=== pg_deeplake stateless init complete ==="
echo "  stateless_enabled = true"
echo "  root_path = ${DEEPLAKE_ROOT_PATH}"
echo "  sync_interval_ms = ${DEEPLAKE_SYNC_INTERVAL_MS:-1000}"
echo "  memory_limit_mb = ${PG_DEEPLAKE_MEMORY_LIMIT_MB:-0}"
echo "  credentials = /dev/shm/deeplake-creds.conf (tmpfs, written by entrypoint)"
