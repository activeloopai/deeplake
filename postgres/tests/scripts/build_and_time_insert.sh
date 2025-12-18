#!/bin/bash

# Build the extension with timing measurements and run insert test

set -e

MAJOR_VERSION=18
MINOR_VERSION=0
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/../../.."

echo "==================================="
echo "Build and Time Insert Test"
echo "==================================="
echo ""

# Step 1: Build the PostgreSQL extension
echo "Step 1: Building PostgreSQL extension..."
echo ""
cd "$PROJECT_ROOT"
python3 scripts/build_pg_ext.py dev --deeplake-shared

if [ $? -ne 0 ]; then
    echo "ERROR: Build failed!"
    exit 1
fi

echo ""
echo "Build completed successfully!"
echo ""

# Step 2: Run the timing test
echo "Step 2: Running insert timing test..."
echo ""
cd "$SCRIPT_DIR"
./time_insert_clean.sh

echo ""
echo "==================================="
echo "Test Complete!"
echo "==================================="
