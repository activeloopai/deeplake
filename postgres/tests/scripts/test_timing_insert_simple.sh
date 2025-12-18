#!/bin/bash

# Simple timing test that works with existing PostgreSQL instance

set -e

MAJOR_VERSION=18
MINOR_VERSION=0
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POSTGRES_SOURCE="$SCRIPT_DIR/../../../cpp/.ext/postgres-REL_${MAJOR_VERSION}_${MINOR_VERSION}"
POSTGRES_INSTALL="$POSTGRES_SOURCE/install"
TESTS_DIR="$SCRIPT_DIR/.."

export LD_LIBRARY_PATH="$POSTGRES_INSTALL/lib:$LD_LIBRARY_PATH"

echo "==================================="
echo "Simple Insert Timing Test"
echo "==================================="
echo ""

# Check if postgres is running
if ! $POSTGRES_INSTALL/bin/pg_isready -q 2>/dev/null; then
    echo "ERROR: PostgreSQL is not running!"
    echo "Please start PostgreSQL first."
    exit 1
fi

echo "PostgreSQL is running - proceeding with test..."
echo ""

cd "$TESTS_DIR"

# Setup schema
echo "Setting up schema..."
$POSTGRES_INSTALL/bin/psql -d postgres > /dev/null 2>&1 <<'EOF'
\i sql/utils.psql
DROP EXTENSION IF EXISTS pg_deeplake CASCADE;
CREATE EXTENSION pg_deeplake;
\i sql/tpch/create_schema.sql
EOF

echo "Schema setup complete!"
echo ""

# Create a log file for timing stats
TIMING_LOG="$TESTS_DIR/logs/timing_stats_$(date +%Y%m%d_%H%M%S).log"
echo "Timing stats will be saved to: $TIMING_LOG"
echo ""

# Run the insert with timing
echo "Running INSERT with timing measurements..."
echo "This may take a while (large dataset)..."
echo ""

# Create a temporary SQL file
TMP_SQL="/tmp/timing_test_$$.sql"
cat > "$TMP_SQL" <<'EOF'
-- Enable timing measurements
SET pg_deeplake.print_runtime_stats = true;
SET client_min_messages = INFO;

-- Show current settings
\echo '=== Current Settings ==='
SHOW pg_deeplake.print_runtime_stats;
SHOW pg_deeplake.use_deeplake_executor;
\echo ''

\timing on

\echo '=== Running INSERT Test ==='
\i sql/tpch/lineitem_2.sql

\echo ''
\echo '=== Test Complete ==='
EOF

# Run psql with the SQL file
$POSTGRES_INSTALL/bin/psql -d postgres -f "$TMP_SQL" 2>&1 | tee "$TIMING_LOG"

# Clean up
rm -f "$TMP_SQL"

echo ""
echo "==================================="
echo "Timing statistics saved to: $TIMING_LOG"
echo "==================================="
echo ""
echo "=== Timing Summary ==="
grep -E "(Query Planning|Executor|Table Scan|DuckDB|Flush|Time:)" "$TIMING_LOG" | tail -30
echo ""
echo "==================================="
echo "Done!"
echo "==================================="
