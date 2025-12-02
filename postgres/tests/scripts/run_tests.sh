#!/bin/bash

# postgres/tests/scripts/run_tests.sh

MAJOR_VERSION=18
MINOR_VERSION=0
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POSTGRES_SOURCE="$SCRIPT_PATH/../../../cpp/.ext/postgres-REL_${MAJOR_VERSION}_${MINOR_VERSION}"
POSTGRES_INSTALL="$POSTGRES_SOURCE/install"
POSTGRES_DATA="$POSTGRES_SOURCE/data"
if [[ "$(uname)" == "Darwin" ]]; then
    DYNAMIC_LIB_SUFFIX=".dylib"
else
    DYNAMIC_LIB_SUFFIX=".so"
fi
LOG_DIR="logs"
RES_DIR="results"
TEST_LOGFILE="$LOG_DIR/test_$(date +%Y%m%d_%H%M%S).log"

source ../scripts/run_pg_server.sh $MAJOR_VERSION $MINOR_VERSION

if [ -d "$LOG_DIR" ]; then
  rm -rf "$LOG_DIR"
fi
mkdir -p "$LOG_DIR"

if [ -d "$RES_DIR" ]; then
  rm -rf "$RES_DIR"
fi
mkdir -p "$RES_DIR"

# Extension-specific settings
EXTENSION_NAME="pg_deeplake"
EXTENSION_PATH="$SCRIPT_PATH/../../"

export LD_LIBRARY_PATH="$POSTGRES_INSTALL/lib:$LD_LIBRARY_PATH"

# Define color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

check_logs_for_errors() {
    local log_file=$1
    local has_error=0

    # Check for various error patterns
    if grep -i "error:" "$log_file" > /dev/null; then
        echo -e "${RED}Found ERROR in log:${NC}"
        grep -i "error:" "$log_file"
        has_error=1
    fi

    if grep -i "fatal:" "$log_file" > /dev/null; then
        echo -e "${RED}Found FATAL error in log:${NC}"
        grep -i "fatal:" "$log_file"
        has_error=1
    fi

    if grep -i "panic:" "$log_file" > /dev/null; then
        echo -e "${RED}Found PANIC in log:${NC}"
        grep -i "panic:" "$log_file"
        has_error=1
    fi

    return $has_error
}

is_test_disabled() {
    local test_name="$1"
    local disabled_tests="$2"
    
    for disabled in $disabled_tests; do
        if [ "$test_name" = "$disabled" ]; then
            return 0  # Test is disabled
        fi
    done
    return 1  # Test is not disabled
}

run_single_test() {
    local test_file="$1"
    local base_name
    local result_file="$RES_DIR/${base_name}_result.log"
    base_name=$(basename "$test_file" .sql)
    echo -e "${YELLOW}Running test: $base_name${NC}"

    # Run the test and capture all output
    "$POSTGRES_INSTALL/bin/psql" -v ON_ERROR_STOP=1 -d postgres \
        -f "$test_file" > "$result_file" 2>&1

    local psql_exit_code=$?

    # Check for errors in the log
    if ! check_logs_for_errors "$result_file"; then
        echo -e "${RED}✗ Test $base_name failed - found errors in log${NC}"
        echo "See $result_file for details"
        return 1
    fi

    if [ $psql_exit_code -ne 0 ]; then
        echo -e "${RED}✗ Test $base_name failed with exit code $psql_exit_code${NC}"
        echo "See $result_file for details"
        return 1
    fi

    echo -e "${GREEN}✓ Test $base_name passed${NC}"
    return 0
}

trap cleanup EXIT

# Parse command line arguments
DISABLED_TESTS=""
SPECIFIC_TEST=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --disabled-tests)
            DISABLED_TESTS="$2"
            shift 2
            ;;
        *)
            SPECIFIC_TEST="$1"
            shift
            ;;
    esac
done

# Run specific test if provided, otherwise run all tests
failed_tests=0
total_tests=0
skipped_tests=0

if [ -n "$SPECIFIC_TEST" ]; then
    if [ -f "sql/$SPECIFIC_TEST.sql" ]; then
        if is_test_disabled "$SPECIFIC_TEST" "$DISABLED_TESTS"; then
            echo -e "${YELLOW}Test $SPECIFIC_TEST is disabled - skipping${NC}"
            exit 0
        fi
        run_single_test "sql/$SPECIFIC_TEST.sql"
        exit $?
    else
        echo "Test file sql/$SPECIFIC_TEST.sql not found"
        exit 1
    fi
fi

for test_file in sql/*.sql; do
    if [ -f "$test_file" ]; then
        base_name=$(basename "$test_file" .sql)
        total_tests=$((total_tests + 1))
        
        if is_test_disabled "$base_name" "$DISABLED_TESTS"; then
            echo -e "${YELLOW}Skipping disabled test: $base_name${NC}"
            skipped_tests=$((skipped_tests + 1))
            continue
        fi
        
        if ! run_single_test "$test_file"; then
            failed_tests=$((failed_tests + 1))
        fi
    fi
done

echo "Test Summary:"
echo "Total tests: $total_tests"
echo "Skipped tests: $skipped_tests"
echo "Failed tests: $failed_tests"
echo "Passed tests: $((total_tests - failed_tests - skipped_tests))"

if [ $failed_tests -gt 0 ]; then
    echo "Some tests failed. Check logs in results/"
    exit 1
fi

echo "All tests passed!"