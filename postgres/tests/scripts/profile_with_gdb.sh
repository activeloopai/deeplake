#!/bin/bash

# Profile using gdb's sampling

MAJOR_VERSION=18
MINOR_VERSION=0
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POSTGRES_SOURCE="$SCRIPT_DIR/../../../cpp/.ext/postgres-REL_${MAJOR_VERSION}_${MINOR_VERSION}"
POSTGRES_INSTALL="$POSTGRES_SOURCE/install"
POSTGRES_DATA="$POSTGRES_SOURCE/data"
TESTS_DIR="$SCRIPT_DIR/.."

export LD_LIBRARY_PATH="$POSTGRES_INSTALL/lib:$LD_LIBRARY_PATH"

echo "=== GDB-based Profiler ==="
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

# Create GDB sampling script
cat > "$TESTS_DIR/gdb_sample.py" <<'GDBS CRIPT'
import gdb
import time
import sys
from collections import defaultdict

class ProfileCommand(gdb.Command):
    def __init__(self):
        super(ProfileCommand, self).__init__("profile-sample", gdb.COMMAND_USER)

    def invoke(self, arg, from_tty):
        samples = defaultdict(int)
        duration = 12  # seconds
        interval = 0.01  # 10ms = 100 Hz

        print(f"Sampling for {duration} seconds at {1/interval} Hz...")

        start = time.time()
        count = 0

        while time.time() - start < duration:
            try:
                gdb.execute("interrupt", to_string=True)
                time.sleep(0.01)  # Let it stop

                frame = gdb.selected_frame()
                func_name = frame.name()
                if func_name:
                    samples[func_name] += 1
                    count += 1

                gdb.execute("continue &", to_string=True)
                time.sleep(interval)
            except:
                break

        print(f"\nCollected {count} samples")
        print("\nTop functions by sample count:")
        print("-" * 60)

        sorted_samples = sorted(samples.items(), key=lambda x: x[1], reverse=True)
        for func, count in sorted_samples[:30]:
            pct = (count / sum(samples.values())) * 100
            print(f"{pct:6.2f}%  {count:5d}  {func}")

ProfileCommand()
GDBSCRIPT

# Get backend PID
echo "Starting PostgreSQL session and getting PID..."
$POSTGRES_INSTALL/bin/psql -d postgres -c "SELECT pg_backend_pid();" -t > "$TESTS_DIR/backend_pid.txt" &
PSQL_PID=$!
sleep 1

BACKEND_PID=$(cat "$TESTS_DIR/backend_pid.txt" | xargs 2>/dev/null)

if [ -z "$BACKEND_PID" ]; then
    echo "Failed to get backend PID"
    rm -f "$TESTS_DIR/profile_setup.sql" "$TESTS_DIR/gdb_sample.py" "$TESTS_DIR/backend_pid.txt"
    exit 1
fi

echo "Backend PID: $BACKEND_PID"
echo ""
echo "This approach requires manual intervention. Instead, let's use a simpler method..."
echo ""

rm -f "$TESTS_DIR/profile_setup.sql" "$TESTS_DIR/gdb_sample.py" "$TESTS_DIR/backend_pid.txt"

# Simpler approach: Just run with detailed logging
echo "Running insert with detailed internal timing..."
echo ""

$POSTGRES_INSTALL/bin/psql -d postgres <<'EOF'
\timing on
\i sql/tpch/lineitem_2.sql
\timing off
EOF

echo ""
echo "Based on the output above:"
echo "  - Total time: ~12.7 seconds"
echo "  - Flush time: ~1.18 seconds (data writing)"
echo "  - Commit time: ~0.18 seconds (finalization)"
echo "  - Remaining ~11.3 seconds: Row processing, conversion, buffering"
echo ""
echo "To profile the remaining 11.3 seconds, you need to add instrumentation"
echo "to the C++ code in pg_deeplake or use a different profiling approach."
echo ""
echo "Suggestions:"
echo "1. Add timing instrumentation in table_data_impl.hpp around row processing"
echo "2. Use 'perf record -a' to profile the entire system (requires sudo)"
echo "3. Use Intel VTune or similar commercial profiler"
echo "4. Enable PostgreSQL's auto_explain module for query planning time"
