#!/bin/bash

# Simple profiling: restart postgres under perf, run insert, analyze

set -e

MAJOR_VERSION=18
MINOR_VERSION=0
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POSTGRES_SOURCE="$SCRIPT_DIR/../../../cpp/.ext/postgres-REL_${MAJOR_VERSION}_${MINOR_VERSION}"
POSTGRES_INSTALL="$POSTGRES_SOURCE/install"
POSTGRES_DATA="$POSTGRES_SOURCE/data"
TESTS_DIR="$SCRIPT_DIR/.."

export LD_LIBRARY_PATH="$POSTGRES_INSTALL/lib:$LD_LIBRARY_PATH"

echo "=== Simple Insert Profiler ==="
echo ""

# Stop any running postgres
echo "Stopping any existing PostgreSQL..."
$POSTGRES_INSTALL/bin/pg_ctl stop -D $POSTGRES_DATA -m fast 2>/dev/null || true
sleep 2

# Start postgres in background under perf
echo "Starting PostgreSQL under perf..."
cd "$TESTS_DIR"

perf record -F 999 -g --call-graph dwarf -o perf.data \
  $POSTGRES_INSTALL/bin/postgres -D $POSTGRES_DATA > logs/pg_perf.log 2>&1 &

PG_PID=$!
echo "Postgres PID: $PG_PID"

# Wait for it to be ready
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
cat > setup.sql <<'EOF'
\i sql/utils.psql
DROP EXTENSION IF EXISTS pg_deeplake CASCADE;
CREATE EXTENSION pg_deeplake;
\i sql/tpch/create_schema.sql
EOF

$POSTGRES_INSTALL/bin/psql -d postgres -f setup.sql > /dev/null 2>&1
rm -f setup.sql

# Run the insert
echo ""
echo "Running insert (this will take ~12 seconds)..."
echo ""

$POSTGRES_INSTALL/bin/psql -d postgres <<'EOF'
\timing on
\i sql/tpch/lineitem_2.sql
\timing off
EOF

# Stop postgres (this will stop perf too)
echo ""
echo "Stopping PostgreSQL..."
$POSTGRES_INSTALL/bin/pg_ctl stop -D $POSTGRES_DATA -m fast

# Wait for perf to finish writing
sleep 2

echo ""
echo "=== Profiling Complete ==="
echo ""

# Analyze
if [ -f "perf.data" ]; then
    SIZE=$(stat -c%s perf.data)
    echo "Perf data: $(numfmt --to=iec-i --suffix=B $SIZE)"

    SAMPLES=$(perf report -i perf.data --stdio 2>/dev/null | grep "^# Samples:" | head -1 | awk '{print $3}')
    echo "Samples: $SAMPLES"
    echo ""

    if [ -z "$SAMPLES" ] || [ "$SAMPLES" -lt 100 ]; then
        echo "WARNING: Low sample count!"
        echo ""
    fi

    echo "Top hotspots:"
    echo ""
    perf report -i perf.data --stdio -n --percent-limit 0.5 2>/dev/null | head -80

    echo ""
    echo "=== Analysis Commands ==="
    echo ""
    echo "Interactive viewer:"
    echo "  cd $TESTS_DIR && perf report -i perf.data"
    echo ""
    echo "Filter for deeplake:"
    echo "  perf report -i $TESTS_DIR/perf.data --stdio | grep -i deeplake"
    echo ""
else
    echo "ERROR: No perf.data file created"
fi

# Restart postgres normally
echo ""
echo "Restarting PostgreSQL normally..."
source ../scripts/run_pg_server.sh $MAJOR_VERSION $MINOR_VERSION > /dev/null 2>&1

echo ""
