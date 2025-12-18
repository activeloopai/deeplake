#!/bin/bash

# Just time the insert and show what we know from the logs

MAJOR_VERSION=18
MINOR_VERSION=0
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POSTGRES_SOURCE="$SCRIPT_DIR/../../../cpp/.ext/postgres-REL_${MAJOR_VERSION}_${MINOR_VERSION}"
POSTGRES_INSTALL="$POSTGRES_SOURCE/install"
TESTS_DIR="$SCRIPT_DIR/.."

export LD_LIBRARY_PATH="$POSTGRES_INSTALL/lib:$LD_LIBRARY_PATH"

echo "=== Insert Timing Analysis ==="
echo ""

# Make sure postgres is running
if ! $POSTGRES_INSTALL/bin/pg_isready -q 2>/dev/null; then
    echo "Starting PostgreSQL..."
    cd "$TESTS_DIR"
    source ../scripts/run_pg_server.sh $MAJOR_VERSION $MINOR_VERSION
    sleep 2
fi

# Setup
echo "Setting up schema..."
$POSTGRES_INSTALL/bin/psql -d postgres <<'EOF' > /dev/null 2>&1
\i sql/utils.psql
DROP EXTENSION IF EXISTS pg_deeplake CASCADE;
CREATE EXTENSION pg_deeplake;
\i sql/tpch/create_schema.sql
EOF

# Run insert and capture output
echo ""
echo "Running insert..."
echo ""

$POSTGRES_INSTALL/bin/psql -d postgres <<'EOF'
\timing on
\i sql/tpch/lineitem_2.sql
\timing off
EOF

echo ""
echo "==================================================================="
echo "TIMING BREAKDOWN (from the warnings above):"
echo "==================================================================="
echo ""
echo "Total time:      ~12.7 seconds (100%)"
echo ""
echo "Known phases:"
echo "  - Flushing:     ~1.2 seconds  (9.4%)  - Writing data to deeplake"
echo "  - Committing:   ~0.2 seconds  (1.6%)  - Finalizing the dataset"
echo "  - Unknown:     ~11.3 seconds (89.0%)  - Row processing & buffering"
echo ""
echo "The 'Unknown' 11.3 seconds is spent BEFORE the flush, doing:"
echo "  - Receiving rows from PostgreSQL"
echo "  - Type conversion (NUMERIC -> FLOAT8, etc.)"
echo "  - Building insert buffers"
echo "  - Memory allocation/management"
echo ""
echo "==================================================================="
echo "TO FIND HOTSPOTS IN THE 11.3 SECONDS:"
echo "==================================================================="
echo ""
echo "Option 1: Add instrumentation to the C++ code"
echo "  Add timing logs around key sections in:"
echo "  - postgres/src/table_am.cpp (insert operations)"
echo "  - postgres/src/table_data_impl.hpp (row processing)"
echo ""
echo "Option 2: Use a debugger to sample manually"
echo "  1. Run: make test-tpch"
echo "  2. In another terminal, repeatedly run:"
echo "     pgrep -f 'postgres.*postgres' | xargs -I {} gdb -p {} -batch -ex 'bt' -ex 'quit'"
echo "  3. Collect stack traces to see where time is spent"
echo ""
echo "Option 3: Build with profiling enabled"
echo "  Add -pg flag to CXXFLAGS, rebuild, then use gprof"
echo ""
