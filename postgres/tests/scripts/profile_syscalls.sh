#!/bin/bash

# Profile using strace to see what system calls are taking time

MAJOR_VERSION=18
MINOR_VERSION=0
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POSTGRES_SOURCE="$SCRIPT_DIR/../../../cpp/.ext/postgres-REL_${MAJOR_VERSION}_${MINOR_VERSION}"
POSTGRES_INSTALL="$POSTGRES_SOURCE/install"
TESTS_DIR="$SCRIPT_DIR/.."

export LD_LIBRARY_PATH="$POSTGRES_INSTALL/lib:$LD_LIBRARY_PATH"

echo "=== System Call Profiler ==="
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
SELECT pg_backend_pid();
EOF

# Get a new connection and its PID
echo "Getting backend PID..."
BACKEND_PID=$($POSTGRES_INSTALL/bin/psql -d postgres -t -c "SELECT pg_backend_pid();" | xargs)

if [ -z "$BACKEND_PID" ]; then
    echo "ERROR: Could not get backend PID"
    exit 1
fi

echo "Backend PID: $BACKEND_PID"
echo ""
echo "Attaching strace and running insert..."
echo "This will take ~12 seconds..."
echo ""

# Run insert while stracing the backend
(
    sleep 2
    $POSTGRES_INSTALL/bin/psql -d postgres <<'EOF'
\timing on
\i sql/tpch/lineitem_2.sql
\timing off
EOF
) &
INSERT_PID=$!

# Attach strace
strace -c -p $BACKEND_PID 2>"$TESTS_DIR/strace_summary.txt" &
STRACE_PID=$!

# Wait for insert to finish
wait $INSERT_PID

# Stop strace
sleep 1
kill -INT $STRACE_PID 2>/dev/null || true
wait $STRACE_PID 2>/dev/null || true

echo ""
echo "=== System Call Summary ==="
echo ""
cat "$TESTS_DIR/strace_summary.txt"

echo ""
echo "This shows time spent in system calls, but most of the work"
echo "is likely in userspace (CPU-bound operations)."
echo ""
echo "Full output saved to: $TESTS_DIR/strace_summary.txt"
echo ""
