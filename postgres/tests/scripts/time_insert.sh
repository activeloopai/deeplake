#!/bin/bash

# Simple script to time the insert operation and show what's happening

MAJOR_VERSION=18
MINOR_VERSION=0
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POSTGRES_SOURCE="$SCRIPT_DIR/../../../cpp/.ext/postgres-REL_${MAJOR_VERSION}_${MINOR_VERSION}"
POSTGRES_INSTALL="$POSTGRES_SOURCE/install"
POSTGRES_DATA="$POSTGRES_SOURCE/data"
TESTS_DIR="$SCRIPT_DIR/.."

export LD_LIBRARY_PATH="$POSTGRES_INSTALL/lib:$LD_LIBRARY_PATH"

echo "=== Timing Insert Operation ==="
echo ""

# Make sure PostgreSQL is running
if ! $POSTGRES_INSTALL/bin/pg_isready -q 2>/dev/null; then
    echo "PostgreSQL is not running. Starting it..."
    cd "$TESTS_DIR"
    source ../scripts/run_pg_server.sh $MAJOR_VERSION $MINOR_VERSION
    sleep 2
fi

# Setup
cat > "$TESTS_DIR/time_insert_setup.sql" <<'EOF'
\i sql/utils.psql
DROP EXTENSION IF EXISTS pg_deeplake CASCADE;
CREATE EXTENSION pg_deeplake;
\i sql/tpch/create_schema.sql
EOF

echo "Setting up schema..."
$POSTGRES_INSTALL/bin/psql -d postgres -f "$TESTS_DIR/time_insert_setup.sql" > /dev/null 2>&1

# Run the insert and capture detailed timing
echo ""
echo "Running insert with detailed timing..."
echo ""

$POSTGRES_INSTALL/bin/psql -d postgres <<'EOF'
-- Enable detailed logging
SET client_min_messages = WARNING;
SET log_min_duration_statement = 0;

\timing on

-- This is the insert that takes ~12 seconds
\i sql/tpch/lineitem_2.sql

\timing off
EOF

rm -f "$TESTS_DIR/time_insert_setup.sql"

echo ""
echo "The warnings above show internal timing:"
echo "  - 'Flushing X insert rows' = starting the bulk insert"
echo "  - 'Flushed X insert rows in Y seconds' = actual data write time"
echo "  - 'Committed dataset in Y seconds' = commit/finalize time"
echo ""
echo "To find hotspots in the C++ code, you need to build with debug symbols and use gdb or better profiling tools."
