#!/bin/bash

# System-wide perf profiling (requires sudo)

MAJOR_VERSION=18
MINOR_VERSION=0
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POSTGRES_SOURCE="$SCRIPT_DIR/../../../cpp/.ext/postgres-REL_${MAJOR_VERSION}_${MINOR_VERSION}"
POSTGRES_INSTALL="$POSTGRES_SOURCE/install"
POSTGRES_DATA="$POSTGRES_SOURCE/data"
TESTS_DIR="$SCRIPT_DIR/.."

export LD_LIBRARY_PATH="$POSTGRES_INSTALL/lib:$LD_LIBRARY_PATH"

echo "=== System-Wide Perf Profiler ==="
echo ""
echo "This will profile ALL processes on the system during the insert."
echo "Press Ctrl+C if you don't want to continue."
echo ""

# Make sure PostgreSQL is running
if ! $POSTGRES_INSTALL/bin/pg_isready -q 2>/dev/null; then
    echo "PostgreSQL is not running. Starting it..."
    cd "$TESTS_DIR"
    source ../scripts/run_pg_server.sh $MAJOR_VERSION $MINOR_VERSION
    sleep 2
fi

# Setup
cat > "$TESTS_DIR/profile_setup.sql" <<'EOF'
\i sql/utils.psql
DROP EXTENSION IF EXISTS pg_deeplake CASCADE;
CREATE EXTENSION pg_deeplake;
\i sql/tpch/create_schema.sql
EOF

echo "Setting up schema..."
$POSTGRES_INSTALL/bin/psql -d postgres -f "$TESTS_DIR/profile_setup.sql" > /dev/null 2>&1

echo ""
echo "Starting system-wide perf recording..."
echo "The insert will start in 3 seconds..."
echo ""

# Start system-wide perf in background
sudo perf record -F 999 -a -g -o "$TESTS_DIR/perf.data" --call-graph dwarf 2>"$TESTS_DIR/perf_err.log" &
PERF_PID=$!

# Give perf time to start
sleep 3

echo "Running insert..."
$POSTGRES_INSTALL/bin/psql -d postgres <<'EOF'
\timing on
\i sql/tpch/lineitem_2.sql
\timing off
EOF

# Stop perf
sleep 1
sudo kill -SIGINT $PERF_PID
wait $PERF_PID 2>/dev/null || true

echo ""
echo "=== Profiling Complete ==="
echo ""

# Check results
if [ -f "$TESTS_DIR/perf.data" ]; then
    PERF_SIZE=$(stat -c%s "$TESTS_DIR/perf.data")
    echo "Perf data size: $(numfmt --to=iec-i --suffix=B $PERF_SIZE)"

    SAMPLE_COUNT=$(sudo perf report -i "$TESTS_DIR/perf.data" --stdio 2>/dev/null | grep "^# Samples:" | head -1 | awk '{print $3}')
    echo "Samples collected: $SAMPLE_COUNT"
    echo ""

    if [ "$SAMPLE_COUNT" -gt 100 ]; then
        echo "Analyzing postgres-related hotspots..."
        echo ""

        # Filter for postgres process only
        sudo perf report -i "$TESTS_DIR/perf.data" --stdio -n --percent-limit 0.5 --comms postgres > "$TESTS_DIR/perf_report_postgres.txt" 2>/dev/null || true

        echo "Top hotspots in postgres process:"
        echo ""
        sudo perf report -i "$TESTS_DIR/perf.data" --stdio -n --percent-limit 0.5 --comms postgres 2>/dev/null | head -100

        echo ""
        echo "=== Next Steps ==="
        echo ""
        echo "1. Interactive view (filter for postgres):"
        echo "   cd $TESTS_DIR && sudo perf report -i perf.data --comms postgres"
        echo ""
        echo "2. Search for deeplake functions:"
        echo "   sudo perf report -i $TESTS_DIR/perf.data --stdio --comms postgres | grep -i deeplake"
        echo ""
        echo "3. Full postgres report:"
        echo "   less $TESTS_DIR/perf_report_postgres.txt"
        echo ""
    else
        echo "Warning: Very few samples collected"
        if [ -f "$TESTS_DIR/perf_err.log" ]; then
            cat "$TESTS_DIR/perf_err.log"
        fi
    fi
else
    echo "Perf failed"
    if [ -f "$TESTS_DIR/perf_err.log" ]; then
        cat "$TESTS_DIR/perf_err.log"
    fi
fi

rm -f "$TESTS_DIR/profile_setup.sql" "$TESTS_DIR/perf_err.log"
