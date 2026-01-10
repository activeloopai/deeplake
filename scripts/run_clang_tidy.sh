#!/bin/bash
set -e

# Script to run clang-tidy on pg_deeplake source files
# Usage: ./scripts/run_clang_tidy.sh [build_dir]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${1:-${PROJECT_ROOT}/builds/deeplake-pg-dev}"

# Convert BUILD_DIR to absolute path if it's relative
if [[ "$BUILD_DIR" != /* ]]; then
    BUILD_DIR="${PROJECT_ROOT}/${BUILD_DIR}"
fi

echo "Running clang-tidy on cpp/deeplake_pg..."
echo "Build directory: ${BUILD_DIR}"

if [ ! -f "${BUILD_DIR}/compile_commands.json" ]; then
    echo "Error: compile_commands.json not found in ${BUILD_DIR}"
    echo "Please build the project first to generate compile_commands.json"
    exit 1
fi

cd "${PROJECT_ROOT}/cpp/deeplake_pg"

# Create temp directory for parallel output
TEMP_DIR=$(mktemp -d)
trap "rm -rf ${TEMP_DIR}" EXIT

echo "Running clang-tidy in parallel..."

# Run clang-tidy for each file in parallel
for file in *.cpp; do
    (
        OUTPUT=$(clang-tidy --header-filter='.*deeplake_pg.*' -p "$BUILD_DIR" "$file" 2>&1 || true)

        # Count warnings in this file (only in deeplake_pg directory)
        FILE_WARNINGS=$(echo "$OUTPUT" | grep -c "deeplake_pg/.*warning:" || true)
        # Count errors in this file
        FILE_ERRORS=$(echo "$OUTPUT" | grep -c "deeplake_pg/.*error:" || true)

        # Save results to temp file
        echo "$file|$FILE_WARNINGS|$FILE_ERRORS" > "${TEMP_DIR}/${file}.count"

        if [ "$FILE_ERRORS" -gt 0 ]; then
            echo "$OUTPUT" | grep "deeplake_pg/.*error:" > "${TEMP_DIR}/${file}.output"
        elif [ "$FILE_WARNINGS" -gt 0 ]; then
            echo "$OUTPUT" | grep "deeplake_pg/.*warning:" > "${TEMP_DIR}/${file}.output"
        fi
    ) &
done

# Wait for all parallel jobs to complete
wait

echo ""
echo "Processing results..."

WARNINGS=0
ERRORS=0

# Process results in order
for file in *.cpp; do
    if [ -f "${TEMP_DIR}/${file}.count" ]; then
        IFS='|' read -r fname FILE_WARNINGS FILE_ERRORS < "${TEMP_DIR}/${file}.count"

        if [ "$FILE_ERRORS" -gt 0 ]; then
            echo "❌ $file - has $FILE_ERRORS errors"
            cat "${TEMP_DIR}/${file}.output"
            ERRORS=$((ERRORS + FILE_ERRORS))
        elif [ "$FILE_WARNINGS" -gt 0 ]; then
            echo "⚠ $file - has $FILE_WARNINGS warnings"
            cat "${TEMP_DIR}/${file}.output"
            WARNINGS=$((WARNINGS + FILE_WARNINGS))
        else
            echo "✓ $file - no issues"
        fi
    fi
done

echo ""
echo "===================="
echo "Clang-Tidy Summary"
echo "===================="
echo "Total warnings: $WARNINGS"
echo "Total errors: $ERRORS"

if [ $ERRORS -gt 0 ]; then
    echo "❌ Clang-tidy found $ERRORS errors!"
    exit 1
elif [ $WARNINGS -gt 0 ]; then
    echo "⚠ Clang-tidy found $WARNINGS warnings (non-blocking)"
    exit 0
else
    echo "✅ No issues found!"
    exit 0
fi
