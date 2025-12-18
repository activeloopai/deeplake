#!/bin/bash

# Profile test_concurrent_insert_and_index_creation with callgrind

set -e

MAJOR_VERSION=18
MINOR_VERSION=0
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POSTGRES_SOURCE="$SCRIPT_DIR/../../../cpp/.ext/postgres-REL_${MAJOR_VERSION}_${MINOR_VERSION}"
POSTGRES_INSTALL="$POSTGRES_SOURCE/install"
POSTGRES_DATA="$POSTGRES_SOURCE/data"
TESTS_DIR="$SCRIPT_DIR/.."

export LD_LIBRARY_PATH="$POSTGRES_INSTALL/lib:$LD_LIBRARY_PATH"

echo "=== Callgrind Concurrent Insert & Index Profiler ==="
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

# Create logs directory if it doesn't exist
mkdir -p logs

valgrind --tool=callgrind \
  --callgrind-out-file=callgrind.out.%p \
  --dump-instr=yes \
  --collect-jumps=yes \
  --collect-systime=yes \
  $POSTGRES_INSTALL/bin/postgres -D $POSTGRES_DATA > logs/pg_callgrind_concurrent.log 2>&1 &

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

# Setup extension
echo ""
echo "Setting up extension..."
$POSTGRES_INSTALL/bin/psql -d postgres <<'EOF' > /dev/null 2>&1
DROP EXTENSION IF EXISTS pg_deeplake CASCADE;
CREATE EXTENSION pg_deeplake;
EOF

# Run the test
echo ""
echo "Running test_concurrent_insert_and_index_creation under callgrind..."
echo "This will take several minutes (normally ~10 seconds, now 5-15 minutes)"
echo ""

time pytest py_tests/test_concurrent_insert_index.py::test_concurrent_insert_and_index_creation -v -s

TEST_EXIT_CODE=$?

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
echo "Test exit code: $TEST_EXIT_CODE"
echo ""

# Find the callgrind output file(s)
CALLGRIND_FILES=($(ls -t callgrind.out.* 2>/dev/null))

if [ ${#CALLGRIND_FILES[@]} -gt 0 ]; then
    echo "Found ${#CALLGRIND_FILES[@]} callgrind output file(s):"
    for FILE in "${CALLGRIND_FILES[@]}"; do
        SIZE=$(stat -c%s "$FILE")
        echo "  - $FILE ($(numfmt --to=iec-i --suffix=B $SIZE))"
    done
    echo ""

    # Use the most recent file
    CALLGRIND_FILE="${CALLGRIND_FILES[0]}"
    echo "Analyzing most recent file: $CALLGRIND_FILE"
    echo ""

    echo "=== Top Hotspots ==="
    echo ""
    callgrind_annotate --auto=yes --threshold=0.5 "$CALLGRIND_FILE" | head -100

    echo ""
    echo "=== Deeplake Functions ==="
    echo ""
    callgrind_annotate --auto=yes "$CALLGRIND_FILE" | grep -B 3 -A 10 -i deeplake | head -50 || echo "No deeplake functions found in top results"

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
    echo "5. Analyze all output files:"
    for FILE in "${CALLGRIND_FILES[@]}"; do
        echo "   callgrind_annotate --auto=yes $TESTS_DIR/$FILE"
    done
    echo ""
else
    echo "ERROR: No callgrind output file found"
    echo "Check logs/pg_callgrind_concurrent.log for errors"
fi

# Restart postgres normally
echo ""
echo "Restarting PostgreSQL normally..."
source ../scripts/run_pg_server.sh $MAJOR_VERSION $MINOR_VERSION > /dev/null 2>&1

echo "Done!"
