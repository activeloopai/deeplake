#!/bin/bash

# Run insert test without profiling overhead

set -e

MAJOR_VERSION=18
MINOR_VERSION=0
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POSTGRES_SOURCE="$SCRIPT_DIR/../../../cpp/.ext/postgres-REL_${MAJOR_VERSION}_${MINOR_VERSION}"
POSTGRES_INSTALL="$POSTGRES_SOURCE/install"
POSTGRES_DATA="$POSTGRES_SOURCE/data"
TESTS_DIR="$SCRIPT_DIR/.."

export LD_LIBRARY_PATH="$POSTGRES_INSTALL/lib:$LD_LIBRARY_PATH"

echo "=== Insert Test Runner ==="
echo ""

# Stop postgres if running
echo "Stopping PostgreSQL..."
$POSTGRES_INSTALL/bin/pg_ctl stop -D $POSTGRES_DATA -m fast 2>/dev/null || true
sleep 2

# Start postgres normally
echo "Starting PostgreSQL..."

cd "$TESTS_DIR"

$POSTGRES_INSTALL/bin/postgres -D $POSTGRES_DATA > logs/pg_test.log 2>&1 &

PG_PID=$!
echo "Postgres PID: $PG_PID"

# Wait for postgres to be ready
echo "Waiting for PostgreSQL to accept connections..."
for i in {1..30}; do
    if $POSTGRES_INSTALL/bin/pg_isready -q 2>/dev/null; then
        echo "PostgreSQL is ready!"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "ERROR: PostgreSQL failed to start"
        kill $PG_PID 2>/dev/null || true
        exit 1
    fi
    sleep 1
done

# Setup schema
echo ""
echo "Setting up schema..."
$POSTGRES_INSTALL/bin/psql -d postgres <<'EOF' > /dev/null 2>&1
\i sql/utils.psql
DROP EXTENSION IF EXISTS pg_deeplake CASCADE;
CREATE EXTENSION pg_deeplake;
\i sql/tpch/create_schema.sql
EOF

# Run the insert
echo ""
echo "Running insert with timing measurements..."
echo ""

# Create a log file for timing stats
TIMING_LOG="$TESTS_DIR/logs/timing_stats_$(date +%Y%m%d_%H%M%S).log"
echo "Timing stats will be saved to: $TIMING_LOG"

$POSTGRES_INSTALL/bin/psql -d postgres 2>&1 | tee "$TIMING_LOG" <<'EOF'
-- Enable timing measurements
SET pg_deeplake.print_runtime_stats = true;
SET client_min_messages = INFO;

-- Show current settings
SHOW pg_deeplake.print_runtime_stats;

\timing on
\echo ''
\echo '=== First Insert Run ==='
\i sql/tpch/lineitem_2.sql

\echo ''
\echo '=== Second Insert Run ==='
\i sql/tpch/lineitem_2.sql
EOF

echo ""
echo "Timing statistics saved to: $TIMING_LOG"
echo ""
echo "=== Timing Summary ==="
grep -E "(Query Planning|Executor|Table Scan|DuckDB|Flush|Time:)" "$TIMING_LOG" | tail -40

# Stop postgres
echo ""
echo "Stopping PostgreSQL..."
$POSTGRES_INSTALL/bin/pg_ctl stop -D $POSTGRES_DATA -m fast

echo ""
echo "=== Test Complete ==="
echo ""

# Restart postgres normally
echo "Restarting PostgreSQL..."
source ../scripts/run_pg_server.sh $MAJOR_VERSION $MINOR_VERSION > /dev/null 2>&1

echo "Done!"
