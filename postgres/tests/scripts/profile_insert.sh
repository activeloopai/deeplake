#!/bin/bash

# postgres/tests/scripts/profile_insert.sh
# Profile the lineitem insert operation to find hotspots

set -e

MAJOR_VERSION=18
MINOR_VERSION=0
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POSTGRES_SOURCE="$SCRIPT_DIR/../../../cpp/.ext/postgres-REL_${MAJOR_VERSION}_${MINOR_VERSION}"
POSTGRES_INSTALL="$POSTGRES_SOURCE/install"
POSTGRES_DATA="$POSTGRES_SOURCE/data"
TESTS_DIR="$SCRIPT_DIR/.."

export LD_LIBRARY_PATH="$POSTGRES_INSTALL/lib:$LD_LIBRARY_PATH"

# Define color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}=== Insert Operation Profiler ===${NC}"
echo ""

# Check kernel settings
PARANOID_LEVEL=$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo "2")
if [ "$PARANOID_LEVEL" -gt 1 ]; then
    echo -e "${YELLOW}Note: perf_event_paranoid is set to $PARANOID_LEVEL${NC}"
    echo "For better profiling, run:"
    echo "  sudo sysctl -w kernel.perf_event_paranoid=1"
    echo ""
fi

# Make sure PostgreSQL is running
if ! $POSTGRES_INSTALL/bin/pg_isready -q 2>/dev/null; then
    echo -e "${YELLOW}PostgreSQL is not running. Starting it...${NC}"
    cd "$TESTS_DIR"
    source ../scripts/run_pg_server.sh $MAJOR_VERSION $MINOR_VERSION
    sleep 2
fi

# Create setup SQL (schema creation)
cat > "$TESTS_DIR/profile_insert_setup.sql" <<'EOF'
\i sql/utils.psql
DROP EXTENSION IF EXISTS pg_deeplake CASCADE;
CREATE EXTENSION pg_deeplake;
\i sql/tpch/create_schema.sql
EOF

echo -e "${YELLOW}Setting up schema...${NC}"
$POSTGRES_INSTALL/bin/psql -d postgres -f "$TESTS_DIR/profile_insert_setup.sql" > /dev/null 2>&1

# Get backend PID in a separate connection that stays open
echo -e "${YELLOW}Starting monitoring session...${NC}"

# Create a script that will hold the connection and run the insert
cat > "$TESTS_DIR/profile_insert_with_monitoring.sh" <<'INNER_SCRIPT'
#!/bin/bash
POSTGRES_INSTALL="$1"
TESTS_DIR="$2"

# Get PID and write to file, then sleep to keep connection alive
$POSTGRES_INSTALL/bin/psql -d postgres -c "SELECT pg_backend_pid();" -t | xargs > "$TESTS_DIR/backend_pid.txt"

# Now run the actual insert while perf is attached
echo "Running insert..."
$POSTGRES_INSTALL/bin/psql -d postgres <<EOF
\timing on
\i sql/tpch/lineitem_2.sql
\timing off
EOF
INNER_SCRIPT

chmod +x "$TESTS_DIR/profile_insert_with_monitoring.sh"

# Start the monitoring script in background
"$TESTS_DIR/profile_insert_with_monitoring.sh" "$POSTGRES_INSTALL" "$TESTS_DIR" &
MONITOR_PID=$!

# Wait for PID file to be created
for i in {1..10}; do
    if [ -f "$TESTS_DIR/backend_pid.txt" ]; then
        break
    fi
    sleep 0.5
done

BACKEND_PID=$(cat "$TESTS_DIR/backend_pid.txt" 2>/dev/null || echo "")

if [ -z "$BACKEND_PID" ]; then
    echo -e "${RED}Failed to get backend PID${NC}"
    kill $MONITOR_PID 2>/dev/null || true
    exit 1
fi

echo -e "${GREEN}Backend PID: $BACKEND_PID${NC}"
echo ""
echo -e "${YELLOW}Starting perf profiling with high sample rate...${NC}"
echo "This will take ~12 seconds..."
echo ""

# Start perf with much higher sampling rate and ensure it's attached before insert runs
PERF_OUTPUT="$TESTS_DIR/perf.data"
perf record -F 9999 -g -p $BACKEND_PID -o "$PERF_OUTPUT" --call-graph dwarf 2>"$TESTS_DIR/perf_errors.log" &
PERF_PID=$!

# Give perf more time to properly attach
sleep 3

# Signal the monitoring script to continue (it's already running the insert by now)
wait $MONITOR_PID 2>/dev/null || true

# Give perf time to finish writing
sleep 2

# Stop perf
kill -SIGINT $PERF_PID 2>/dev/null || true
wait $PERF_PID 2>/dev/null || true

echo ""
echo -e "${GREEN}=== Profiling Complete ===${NC}"
echo ""

# Clean up
rm -f "$TESTS_DIR/backend_pid.txt" "$TESTS_DIR/profile_insert_with_monitoring.sh"

# Check if perf.data was created and has data
if [ -f "$PERF_OUTPUT" ] && [ -s "$PERF_OUTPUT" ]; then
    echo "Perf data saved to: $PERF_OUTPUT"

    # Check the file size
    PERF_SIZE=$(stat -c%s "$PERF_OUTPUT" 2>/dev/null || echo "0")
    echo "Perf data size: $(numfmt --to=iec-i --suffix=B $PERF_SIZE 2>/dev/null || echo "$PERF_SIZE bytes")"

    # Check number of samples
    SAMPLE_COUNT=$(perf report -i "$PERF_OUTPUT" --stdio 2>/dev/null | grep "^# Samples:" | head -1 | awk '{print $3}')
    echo "Samples collected: $SAMPLE_COUNT"
    echo ""

    if [ "$PERF_SIZE" -lt 50000 ] || [ "$SAMPLE_COUNT" -lt 100 ]; then
        echo -e "${YELLOW}Warning: Low sample count. The process may have been idle or perf attached too late.${NC}"
        if [ -f "$TESTS_DIR/perf_errors.log" ] && [ -s "$TESTS_DIR/perf_errors.log" ]; then
            echo "Perf errors/warnings:"
            cat "$TESTS_DIR/perf_errors.log"
        fi
        echo ""
    fi

    echo -e "${BLUE}Analyzing hotspots...${NC}"
    echo ""

    # Generate report
    perf report -i "$PERF_OUTPUT" --stdio -n --percent-limit 0.5 > "$TESTS_DIR/perf_report.txt" 2>/dev/null || true

    # Show top functions
    echo -e "${YELLOW}Top hotspots (>0.5% of samples):${NC}"
    echo ""
    if perf report -i "$PERF_OUTPUT" --stdio -n --percent-limit 0.5 2>/dev/null | head -100; then
        echo ""
        echo -e "${GREEN}Successfully generated report!${NC}"
    else
        echo -e "${YELLOW}Could not generate text report, but data is available for interactive viewing${NC}"
    fi

    echo ""
    echo -e "${BLUE}=== Next Steps ===${NC}"
    echo ""
    echo "1. View interactive report (BEST option - use arrow keys, Enter to expand, 'a' to annotate):"
    echo -e "   ${YELLOW}cd $TESTS_DIR && perf report -i perf.data${NC}"
    echo ""
    echo "2. View full text report:"
    echo -e "   ${YELLOW}less $TESTS_DIR/perf_report.txt${NC}"
    echo ""
    echo "3. Search for specific functions:"
    echo -e "   ${YELLOW}perf report -i $TESTS_DIR/perf.data --stdio | grep -i 'deeplake\\|flush\\|commit\\|insert'${NC}"
    echo ""
    echo "4. Show line-by-line hotspots for a function (replace FUNCTION with actual name):"
    echo -e "   ${YELLOW}perf annotate -i $TESTS_DIR/perf.data FUNCTION${NC}"
    echo ""
else
    echo -e "${RED}Perf profiling failed or produced no output${NC}"
    if [ -f "$TESTS_DIR/perf_errors.log" ]; then
        echo "Errors:"
        cat "$TESTS_DIR/perf_errors.log"
    fi
    echo ""
    echo "Make sure you ran:"
    echo "  sudo sysctl -w kernel.perf_event_paranoid=1"
    echo "  sudo sysctl -w kernel.kptr_restrict=0"
fi

# Clean up temp files
rm -f "$TESTS_DIR/profile_insert_setup.sql" "$TESTS_DIR/perf_errors.log"

echo ""
