#!/bin/bash

# postgres/tests/scripts/profile_tpch.sh
# Profile TPCH test with callgrind to find hotspots in pg_deeplake and deeplake_api

set -e

MAJOR_VERSION=18
MINOR_VERSION=0
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POSTGRES_SOURCE="$SCRIPT_DIR/../../../cpp/.ext/postgres-REL_${MAJOR_VERSION}_${MINOR_VERSION}"
POSTGRES_INSTALL="$POSTGRES_SOURCE/install"
POSTGRES_DATA="$POSTGRES_SOURCE/data"
TESTS_DIR="$SCRIPT_DIR/.."

export LD_LIBRARY_PATH="$POSTGRES_INSTALL/lib:$LD_LIBRARY_PATH"

# Create logs directory if it doesn't exist
mkdir -p "$TESTS_DIR/logs"

# Define color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}=== TPCH Callgrind Profiler ===${NC}"
echo ""

# Stop existing PostgreSQL
echo -e "${YELLOW}Stopping any existing PostgreSQL instance...${NC}"
$POSTGRES_INSTALL/bin/pg_ctl stop -D $POSTGRES_DATA -m fast 2>/dev/null || true
sleep 2

# Start PostgreSQL under callgrind
echo -e "${YELLOW}Starting PostgreSQL under callgrind...${NC}"
echo "This will be slower than normal execution - be patient!"
echo ""

valgrind --tool=callgrind \
  --callgrind-out-file="$TESTS_DIR/callgrind.out.tpch.%p" \
  --collect-jumps=yes \
  --separate-threads=yes \
  --trace-children=yes \
  --instr-atstart=yes \
  $POSTGRES_INSTALL/bin/postgres -D $POSTGRES_DATA > "$TESTS_DIR/logs/pg_callgrind.log" 2>&1 &

PG_VALGRIND_PID=$!
echo "PostgreSQL started under callgrind (valgrind PID: $PG_VALGRIND_PID)"

# Wait for PostgreSQL to be ready
echo -e "${YELLOW}Waiting for PostgreSQL to be ready...${NC}"
for i in {1..30}; do
    if $POSTGRES_INSTALL/bin/pg_isready -q 2>/dev/null; then
        echo -e "${GREEN}PostgreSQL is ready!${NC}"
        break
    fi
    if [ $i -eq 30 ]; then
        echo -e "${RED}PostgreSQL failed to start. Check logs/pg_callgrind.log${NC}"
        kill $PG_VALGRIND_PID 2>/dev/null || true
        exit 1
    fi
    sleep 1
done

# Run the TPCH test
echo ""
echo -e "${YELLOW}Running TPCH test...${NC}"
echo "Output will be saved to logs/tpch_result.log"
echo ""

if $POSTGRES_INSTALL/bin/psql -d postgres -f "$TESTS_DIR/sql/tpch.sql" > "$TESTS_DIR/logs/tpch_result.log" 2>&1; then
    echo -e "${GREEN}TPCH test completed successfully!${NC}"
else
    echo -e "${RED}TPCH test failed. Check logs/tpch_result.log${NC}"
fi

# Stop PostgreSQL gracefully
echo ""
echo -e "${YELLOW}Stopping PostgreSQL...${NC}"
$POSTGRES_INSTALL/bin/pg_ctl stop -D $POSTGRES_DATA -m fast

# Wait for callgrind to finish writing
echo "Waiting for callgrind to finalize output files..."
wait $PG_VALGRIND_PID 2>/dev/null || true

echo ""
echo -e "${GREEN}=== Profiling Complete ===${NC}"
echo ""
echo "Callgrind output files generated in: $TESTS_DIR/"
ls -lh "$TESTS_DIR"/callgrind.out.tpch.* 2>/dev/null || echo "No callgrind files found"
echo ""
echo -e "${BLUE}To analyze the results:${NC}"
echo ""
echo "1. View summary with annotations:"
echo -e "   ${YELLOW}callgrind_annotate --auto=yes callgrind.out.tpch.* | less${NC}"
echo ""
echo "2. Filter for pg_deeplake functions:"
echo -e "   ${YELLOW}callgrind_annotate --auto=yes callgrind.out.tpch.* | grep -A 20 pg_deeplake${NC}"
echo ""
echo "3. Filter for deeplake_api functions:"
echo -e "   ${YELLOW}callgrind_annotate --auto=yes callgrind.out.tpch.* | grep -A 20 deeplake${NC}"
echo ""
echo "4. Use kcachegrind GUI (if installed):"
echo -e "   ${YELLOW}kcachegrind callgrind.out.tpch.*${NC}"
echo ""
echo "5. Get top 20 hotspots:"
echo -e "   ${YELLOW}callgrind_annotate --threshold=0.1 callgrind.out.tpch.* | head -50${NC}"
echo ""
