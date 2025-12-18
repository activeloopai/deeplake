#!/bin/bash

# Profile just the insert with callgrind

set -e

MAJOR_VERSION=18
MINOR_VERSION=0
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POSTGRES_SOURCE="$SCRIPT_DIR/../../../cpp/.ext/postgres-REL_${MAJOR_VERSION}_${MINOR_VERSION}"
POSTGRES_INSTALL="$POSTGRES_SOURCE/install"
POSTGRES_DATA="$POSTGRES_SOURCE/data"
TESTS_DIR="$SCRIPT_DIR/.."

export LD_LIBRARY_PATH="$POSTGRES_INSTALL/lib:$LD_LIBRARY_PATH"

echo "=== Callgrind Insert Profiler ==="
echo ""

# Check if valgrind is installed
if ! command -v valgrind &> /dev/null; then
    echo "ERROR: valgrind not found. Install it with:"
    echo "  sudo apt-get install valgrind"
    exit 1
fi

# Stop postgres if running
echo "Stopping PostgreSQL..."
$POSTGRES_INSTALL/bin/pg_ctl stop -D $POSTGRES_DATA -m fast 2>/dev/null || true
sleep 2

# Start postgres under callgrind
echo "Starting PostgreSQL under callgrind..."
echo "NOTE: This will be VERY slow (20-50x slower than normal)"
echo ""

cd "$TESTS_DIR"

valgrind --tool=callgrind \
  --callgrind-out-file=callgrind.out.%p \
  --dump-instr=yes \
  --collect-jumps=yes \
  --collect-systime=yes \
  $POSTGRES_INSTALL/bin/postgres -D $POSTGRES_DATA > logs/pg_callgrind.log 2>&1 &

PG_PID=$!
echo "Postgres (under valgrind) PID: $PG_PID"

# Wait for postgres to be ready
echo "Waiting for PostgreSQL to accept connections (this takes longer under callgrind)..."
for i in {1..60}; do
    if $POSTGRES_INSTALL/bin/pg_isready -q 2>/dev/null; then
        echo "PostgreSQL is ready!"
        break
    fi
    if [ $i -eq 60 ]; then
        echo "ERROR: PostgreSQL failed to start"
        kill $PG_PID 2>/dev/null || true
        exit 1
    fi
    sleep 2
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
echo "Running insert under callgrind..."
echo "This will take several minutes (normally 12 seconds, now 5-10 minutes)"
echo ""

time $POSTGRES_INSTALL/bin/psql -d postgres <<'EOF'
\i sql/tpch/lineitem_2.sql
EOF

# Stop postgres
echo ""
echo "Stopping PostgreSQL..."
$POSTGRES_INSTALL/bin/pg_ctl stop -D $POSTGRES_DATA -m fast

# Wait for callgrind to finish writing
echo "Waiting for callgrind to finalize..."
sleep 5

echo ""
echo "=== Profiling Complete ==="
echo ""

# Find the callgrind output file
CALLGRIND_FILE=$(ls -t callgrind.out.* 2>/dev/null | head -1)

if [ -f "$CALLGRIND_FILE" ]; then
    SIZE=$(stat -c%s "$CALLGRIND_FILE")
    echo "Callgrind output: $CALLGRIND_FILE"
    echo "Size: $(numfmt --to=iec-i --suffix=B $SIZE)"
    echo ""

    echo "Analyzing top hotspots..."
    echo ""
    callgrind_annotate --auto=yes --threshold=0.5 "$CALLGRIND_FILE" | head -100

    echo ""
    echo "=== Analysis Commands ==="
    echo ""
    echo "1. View full annotated report:"
    echo "   callgrind_annotate --auto=yes $TESTS_DIR/$CALLGRIND_FILE | less"
    echo ""
    echo "2. Filter for deeplake functions:"
    echo "   callgrind_annotate --auto=yes $TESTS_DIR/$CALLGRIND_FILE | grep -B 3 -A 10 -i deeplake"
    echo ""
    echo "3. Interactive GUI (if installed):"
    echo "   kcachegrind $TESTS_DIR/$CALLGRIND_FILE"
    echo ""
    echo "4. See specific function costs:"
    echo "   callgrind_annotate --include=<path> $TESTS_DIR/$CALLGRIND_FILE"
    echo ""
else
    echo "ERROR: No callgrind output file found"
    echo "Check logs/pg_callgrind.log for errors"
fi

# Restart postgres normally
echo ""
echo "Restarting PostgreSQL normally..."
source ../scripts/run_pg_server.sh $MAJOR_VERSION $MINOR_VERSION > /dev/null 2>&1

echo "Done!"
