#!/bin/bash
set -e

# Script to run clang-tidy on pg_deeplake source files
# Usage: ./scripts/run_clang_tidy.sh [build_dir]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${1:-${PROJECT_ROOT}/builds/deeplake-pg-dev}"

echo "Running clang-tidy on cpp/deeplake_pg..."
echo "Build directory: ${BUILD_DIR}"

if [ ! -f "${BUILD_DIR}/compile_commands.json" ]; then
    echo "Error: compile_commands.json not found in ${BUILD_DIR}"
    echo "Please build the project first to generate compile_commands.json"
    exit 1
fi

cd "${PROJECT_ROOT}/cpp/deeplake_pg"

WARNINGS=0
ERRORS=0

for file in *.cpp; do
    echo "Checking $file..."
    OUTPUT=$(clang-tidy --header-filter='.*deeplake_pg.*' -p "$BUILD_DIR" "$file" 2>&1 || true)

    # Count warnings in this file (only in deeplake_pg directory)
    FILE_WARNINGS=$(echo "$OUTPUT" | grep -c "deeplake_pg/.*warning:" || true)
    # Count errors in this file
    FILE_ERRORS=$(echo "$OUTPUT" | grep -c "deeplake_pg/.*error:" || true)

    if [ "$FILE_ERRORS" -gt 0 ]; then
        echo "❌ $file - has $FILE_ERRORS errors"
        echo "$OUTPUT" | grep "deeplake_pg/.*error:"
        ERRORS=$((ERRORS + FILE_ERRORS))
    elif [ "$FILE_WARNINGS" -gt 0 ]; then
        echo "⚠ $file - has $FILE_WARNINGS warnings"
        echo "$OUTPUT" | grep "deeplake_pg/.*warning:"
        WARNINGS=$((WARNINGS + FILE_WARNINGS))
    else
        echo "✓ $file - no issues"
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
