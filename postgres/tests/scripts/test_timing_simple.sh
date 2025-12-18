#!/bin/bash

# Simple test to verify timing measurements are working

set -e

MAJOR_VERSION=18
MINOR_VERSION=0
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POSTGRES_SOURCE="$SCRIPT_DIR/../../../cpp/.ext/postgres-REL_${MAJOR_VERSION}_${MINOR_VERSION}"
POSTGRES_INSTALL="$POSTGRES_SOURCE/install"

export LD_LIBRARY_PATH="$POSTGRES_INSTALL/lib:$LD_LIBRARY_PATH"

echo "==================================="
echo "Simple Timing Test"
echo "==================================="
echo ""

# Check if postgres is running
if ! $POSTGRES_INSTALL/bin/pg_isready -q 2>/dev/null; then
    echo "ERROR: PostgreSQL is not running!"
    echo "Please start PostgreSQL first:"
    echo "  cd $SCRIPT_DIR/.."
    echo "  source ../scripts/run_pg_server.sh $MAJOR_VERSION $MINOR_VERSION"
    exit 1
fi

echo "Running simple query with timing measurements..."
echo ""

$POSTGRES_INSTALL/bin/psql -d postgres <<'EOF'
-- Enable timing measurements
SET pg_deeplake.print_runtime_stats = true;
SET client_min_messages = INFO;

-- Show current settings
\echo '=== Current Settings ==='
SHOW pg_deeplake.print_runtime_stats;
SHOW pg_deeplake.use_deeplake_executor;
\echo ''

\timing on

-- Simple test query
\echo '=== Test Query 1: Simple SELECT ==='
SELECT 1 as test;

\echo ''
\echo '=== Test Query 2: Table Query (if lineitem exists) ==='
SELECT COUNT(*) FROM lineitem LIMIT 1;

\echo ''
\echo 'If you see timing messages (e.g., "Query Planning: X.XX ms"), timing is working!'
\echo 'If not, the extension may need to be rebuilt.'
EOF

echo ""
echo "==================================="
echo "Test Complete!"
echo "==================================="
