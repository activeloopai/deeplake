#!/bin/bash

# Profile the PostgreSQL backend process with Intel VTune

set -e

MAJOR_VERSION=18
MINOR_VERSION=0
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POSTGRES_SOURCE="$SCRIPT_DIR/../../../cpp/.ext/postgres-REL_${MAJOR_VERSION}_${MINOR_VERSION}"
POSTGRES_INSTALL="$POSTGRES_SOURCE/install"
POSTGRES_DATA="$POSTGRES_SOURCE/data"
TESTS_DIR="$SCRIPT_DIR/.."

export LD_LIBRARY_PATH="$POSTGRES_INSTALL/lib:$LD_LIBRARY_PATH"

echo "=== VTune Insert Profiler ==="
echo ""

# Check if vtune is available
if ! command -v vtune &> /dev/null; then
    echo "ERROR: vtune command not found"
    echo "Make sure VTune is installed and in PATH:"
    echo "  source /opt/intel/oneapi/vtune/latest/env/vars.sh"
    exit 1
fi

echo "VTune found: $(which vtune)"
echo ""

# Check and fix kernel settings for VTune
PTRACE_SCOPE=$(cat /proc/sys/kernel/yama/ptrace_scope 2>/dev/null || echo "0")
if [ "$PTRACE_SCOPE" != "0" ]; then
    echo "Setting ptrace_scope to 0 for VTune..."
    sudo sysctl -w kernel.yama.ptrace_scope=0
fi

PERF_PARANOID=$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo "2")
if [ "$PERF_PARANOID" != "-1" ]; then
    echo "Setting perf_event_paranoid to -1 for VTune..."
    sudo sysctl -w kernel.perf_event_paranoid=-1
fi

KPTR_RESTRICT=$(cat /proc/sys/kernel/kptr_restrict 2>/dev/null || echo "0")
if [ "$KPTR_RESTRICT" != "0" ]; then
    echo "Setting kptr_restrict to 0..."
    sudo sysctl -w kernel.kptr_restrict=0
fi

echo ""

# Make sure postgres is running
if ! $POSTGRES_INSTALL/bin/pg_isready -q 2>/dev/null; then
    echo "Starting PostgreSQL..."
    cd "$TESTS_DIR"
    source ../scripts/run_pg_server.sh $MAJOR_VERSION $MINOR_VERSION
    sleep 2
fi

# Setup schema
echo "Setting up schema..."
$POSTGRES_INSTALL/bin/psql -d postgres <<'EOF' > /dev/null 2>&1
\i sql/utils.psql
DROP EXTENSION IF EXISTS pg_deeplake CASCADE;
CREATE EXTENSION pg_deeplake;
\i sql/tpch/create_schema.sql
EOF

# Create VTune result directory
VTUNE_RESULT_DIR="$TESTS_DIR/vtune_results"
rm -rf "$VTUNE_RESULT_DIR"

echo ""
echo "Starting VTune collection on ALL postgres processes..."
echo "This will capture any postgres backend that runs during the insert"
echo ""

# Get the main postmaster PID to filter its children
POSTMASTER_PID=$(pgrep -f "postgres -D" | head -1)

if [ -z "$POSTMASTER_PID" ]; then
    echo "ERROR: Could not find postmaster process"
    exit 1
fi

echo "Postmaster PID: $POSTMASTER_PID"

# Create a script that runs the insert
cat > "$TESTS_DIR/run_insert_only.sh" <<INSERTSCRIPT
#!/bin/bash
export LD_LIBRARY_PATH="$POSTGRES_INSTALL/lib:\$LD_LIBRARY_PATH"
cd "$TESTS_DIR"
$POSTGRES_INSTALL/bin/psql -d postgres <<'EOF'
\timing on
\i sql/tpch/lineitem_2.sql
\timing off
EOF
INSERTSCRIPT

chmod +x "$TESTS_DIR/run_insert_only.sh"

# Run VTune profiling the script
# VTune will automatically follow child processes (postgres backends)
echo "Running VTune collection..."
echo ""

vtune -collect hotspots \
  -knob sampling-mode=sw \
  -knob enable-stack-collection=true \
  -follow-child \
  -result-dir "$VTUNE_RESULT_DIR" \
  -- "$TESTS_DIR/run_insert_only.sh"

echo ""
echo "=== Profiling Complete ==="
echo ""

# Clean up
rm -f "$TESTS_DIR/run_insert_only.sh"

# Generate report
if [ -d "$VTUNE_RESULT_DIR" ]; then
    echo "VTune results saved to: $VTUNE_RESULT_DIR"
    echo ""

    echo "Generating hotspots report..."
    echo ""

    # Generate top hotspots report
    vtune -report hotspots \
      -result-dir "$VTUNE_RESULT_DIR" \
      -format text \
      -report-output "$TESTS_DIR/vtune_hotspots.txt" \
      2>/dev/null || true

    if [ -f "$TESTS_DIR/vtune_hotspots.txt" ]; then
        echo "=== Top Hotspots ==="
        echo ""
        head -150 "$TESTS_DIR/vtune_hotspots.txt"
        echo ""
        echo "==================================================================="
        echo ""
    fi

    # Try to generate a report grouped by process/module
    echo "Generating module-level report..."
    vtune -report hotspots \
      -group-by module \
      -result-dir "$VTUNE_RESULT_DIR" \
      -format text \
      -report-output "$TESTS_DIR/vtune_modules.txt" \
      2>/dev/null || true

    if [ -f "$TESTS_DIR/vtune_modules.txt" ]; then
        echo ""
        echo "=== Top Modules ==="
        echo ""
        head -50 "$TESTS_DIR/vtune_modules.txt"
        echo ""
    fi

    echo "=== Analysis Commands ==="
    echo ""
    echo "1. View full text report:"
    echo "   less $TESTS_DIR/vtune_hotspots.txt"
    echo ""
    echo "2. View modules report:"
    echo "   less $TESTS_DIR/vtune_modules.txt"
    echo ""
    echo "3. Filter for deeplake functions:"
    echo "   grep -i deeplake $TESTS_DIR/vtune_hotspots.txt"
    echo ""
    echo "4. Filter for postgres process only:"
    echo "   vtune -report hotspots -filter process=postgres -result-dir $VTUNE_RESULT_DIR"
    echo ""
    echo "5. View call stacks (top-down):"
    echo "   vtune -report top-down -result-dir $VTUNE_RESULT_DIR | less"
    echo ""
    echo "6. Open in VTune GUI (RECOMMENDED):"
    echo "   vtune-gui $VTUNE_RESULT_DIR"
    echo ""
    echo "   In the GUI, filter by process name 'postgres' to see only backend data"
    echo ""
else
    echo "ERROR: No VTune results found"
fi

echo ""
