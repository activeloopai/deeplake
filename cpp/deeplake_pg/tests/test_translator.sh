#!/bin/bash

# Standalone build and test script for PostgreSQL to DuckDB translator
# This script is compiler-agnostic and will auto-detect available compilers

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "Building translator test..."

# Detect available C++ compiler
CXX_COMPILER=""
if command -v g++ &> /dev/null; then
    CXX_COMPILER="g++"
    echo "Using g++ compiler"
elif command -v clang++ &> /dev/null; then
    CXX_COMPILER="clang++"
    echo "Using clang++ compiler"
elif command -v c++ &> /dev/null; then
    CXX_COMPILER="c++"
    echo "Using default c++ compiler"
else
    echo -e "${RED}Error: No C++ compiler found (g++, clang++, or c++)${NC}"
    exit 1
fi

# Detect C++ standard support
CXX_STANDARD="-std=c++20"
if ! $CXX_COMPILER $CXX_STANDARD --version &> /dev/null; then
    echo "C++20 not supported, falling back to C++17"
    CXX_STANDARD="-std=c++17"
fi

# Build the test
$CXX_COMPILER $CXX_STANDARD -o translator_test \
    ../pg_to_duckdb_translator.cpp \
    pg_to_duckdb_translator_test.cpp \
    -I.. \
    -Wall -Wextra 2>&1 | grep -v "warning:" || true

if [ ! -f translator_test ]; then
    echo -e "${RED}Build failed!${NC}"
    exit 1
fi

echo -e "${GREEN}Build successful!${NC}"
echo ""
echo "Running translator tests..."
echo ""

# Run the tests and capture output
TEST_OUTPUT=$(./translator_test 2>&1)
EXIT_CODE=$?

# Display the output
echo "$TEST_OUTPUT"

# Check if all tests passed
if [ $EXIT_CODE -eq 0 ] && echo "$TEST_OUTPUT" | grep -q "All tests completed"; then
    PASSED=$(echo "$TEST_OUTPUT" | grep -c "✓ PASSED" || true)
    FAILED=$(echo "$TEST_OUTPUT" | grep -c "✗ FAILED" || true)

    echo ""
    echo "======================================"
    if [ $FAILED -eq 0 ]; then
        echo -e "${GREEN}All $PASSED tests PASSED!${NC}"
    else
        echo -e "${YELLOW}Results: $PASSED passed, $FAILED failed${NC}"
    fi
    echo "======================================"
else
    echo ""
    echo -e "${RED}Test execution failed!${NC}"
    exit 1
fi

# Cleanup
rm -f translator_test

echo ""
echo "Done!"
