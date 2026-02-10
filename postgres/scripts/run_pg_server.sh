#!/bin/bash

set -eo pipefail

MAJOR_VERSION="${1:-18}"
MINOR_VERSION="${2:-0}"
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POSTGRES_SOURCE="$SCRIPT_PATH/../../cpp/.ext/postgres-REL_${MAJOR_VERSION}_${MINOR_VERSION}"
POSTGRES_INSTALL="$POSTGRES_SOURCE/install"
POSTGRES_DATA="$POSTGRES_SOURCE/data"
if [[ "$(uname)" == "Darwin" ]]; then
    DYNAMIC_LIB_SUFFIX=".dylib"
else
    DYNAMIC_LIB_SUFFIX=".so"
fi
LOG_DIR="logs"
RES_DIR="results"
TEST_LOGFILE="$LOG_DIR/test_$(date +%Y%m%d_%H%M%S).log"

if [ -d "$LOG_DIR" ]; then
  rm -rf "$LOG_DIR"
fi
mkdir -p "$LOG_DIR"

if [ -d "$RES_DIR" ]; then
  rm -rf "$RES_DIR"
fi
mkdir -p "$RES_DIR"

# Extension-specific settings
EXTENSION_NAME="pg_deeplake"
EXTENSION_PATH="$SCRIPT_PATH/.."

export LD_LIBRARY_PATH="$POSTGRES_INSTALL/lib:$LD_LIBRARY_PATH"

# Helper functions
is_postgres_running() {
    "$POSTGRES_INSTALL/bin/pg_ctl" status -D "$POSTGRES_DATA" > /dev/null 2>&1
}

stop_postgres() {
    if is_postgres_running; then
        echo "Stopping PostgreSQL..."
        "$POSTGRES_INSTALL/bin/pg_ctl" stop -D "$POSTGRES_DATA" -m fast
        sleep 2
    fi
}

cleanup() {
    stop_postgres
    echo "Cleanup complete"
}

install_extension() {
    # Create extension directory if it doesn't exist
    local ext_dir="$POSTGRES_INSTALL/share/extension"
    local lib_dir="$POSTGRES_INSTALL/lib"
    mkdir -p "$ext_dir"

    # Copy extension files
    if [ -d "$EXTENSION_PATH" ]; then
        echo "Installing extension files from $EXTENSION_PATH to $ext_dir"
        cp -f "$EXTENSION_PATH"/*.control "$ext_dir"/ || return 1
        cp -f "$EXTENSION_PATH"/*.sql "$ext_dir"/ || return 1
        cp -f "$EXTENSION_PATH"/"$EXTENSION_NAME"_"$MAJOR_VERSION""$DYNAMIC_LIB_SUFFIX" "$lib_dir"/"$EXTENSION_NAME" || return 1
    else
        echo "Error: Extension directory $EXTENSION_PATH not found"
        return 1
    fi
}

# Stop any existing postgres instance
stop_postgres

# Remove existing data directory if it exists
if [ -d "$POSTGRES_DATA" ]; then
    rm -rf "$POSTGRES_DATA"
fi

install_extension
# Initialize database with 'postgres' superuser (matches Docker POSTGRES_USER=postgres)
"$POSTGRES_INSTALL/bin/initdb" -D "$POSTGRES_DATA" -U postgres
echo "shared_preload_libraries = 'pg_deeplake'" >> "$POSTGRES_DATA/postgresql.conf"
echo "max_connections = 300" >> "$POSTGRES_DATA/postgresql.conf"
echo "shared_buffers = 128MB" >> "$POSTGRES_DATA/postgresql.conf"
#echo "log_min_messages = debug1" >> "$POSTGRES_DATA/postgresql.conf"

# Configure pg_hba.conf: use scram-sha-256 for TCP connections (matches Docker behavior)
# Keep trust for local socket connections (for admin convenience)
if [[ "$(uname)" == "Darwin" ]]; then
    sed -i '' 's/^\(host.*all.*all.*127\.0\.0\.1\/32\s*\)trust/\1scram-sha-256/' "$POSTGRES_DATA/pg_hba.conf"
    sed -i '' 's/^\(host.*all.*all.*::1\/128\s*\)trust/\1scram-sha-256/' "$POSTGRES_DATA/pg_hba.conf"
else
    sed -i 's/^\(host.*all.*all.*127\.0\.0\.1\/32\s*\)trust/\1scram-sha-256/' "$POSTGRES_DATA/pg_hba.conf"
    sed -i 's/^\(host.*all.*all.*::1\/128\s*\)trust/\1scram-sha-256/' "$POSTGRES_DATA/pg_hba.conf"
fi

# Start PostgreSQL temporarily to set password
"$POSTGRES_INSTALL/bin/pg_ctl" -D "$POSTGRES_DATA" -l "$TEST_LOGFILE" -t 120 start

# Set postgres password (matches Docker POSTGRES_PASSWORD=password)
"$POSTGRES_INSTALL/bin/psql" -U postgres -c "ALTER USER postgres PASSWORD 'password';"

# Stop PostgreSQL and restart in foreground
stop_postgres

echo "Starting PostgreSQL in foreground..."
exec "$POSTGRES_INSTALL/bin/postgres" -D "$POSTGRES_DATA"
