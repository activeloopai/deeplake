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

# Find all C++ source files
CPP_FILES=$(find . -name "*.cpp" -type f)

if [ -z "$CPP_FILES" ]; then
    echo "No C++ files found in cpp/deeplake_pg"
    exit 1
fi

echo "Found $(echo "$CPP_FILES" | wc -l) C++ files to analyze"

# Run clang-tidy on all files
FAILED=0
WARNING_COUNT=0

for file in $CPP_FILES; do
    echo "Analyzing: $file"
    # Run clang-tidy and capture output
    OUTPUT=$(clang-tidy \
        -p "${BUILD_DIR}" \
        "$file" \
        2>&1 || true)

    # Count warnings in this file (excluding header warnings)
    FILE_WARNINGS=$(echo "$OUTPUT" | grep -c "deeplake_pg/.*warning:" || true)
    WARNING_COUNT=$((WARNING_COUNT + FILE_WARNINGS))

    # Check for errors
    if echo "$OUTPUT" | grep -q "error:"; then
        echo "  ❌ Errors found in $file"
        echo "$OUTPUT" | grep "error:"
        FAILED=1
    elif [ "$FILE_WARNINGS" -gt 0 ]; then
        echo "  ⚠️  $FILE_WARNINGS warnings"
    else
        echo "  ✅ No issues"
    fi
done

echo ""
echo "===================="
echo "Clang-Tidy Summary"
echo "===================="
echo "Total warnings: $WARNING_COUNT"

if [ $FAILED -ne 0 ]; then
    echo "❌ Clang-tidy found errors!"
    exit 1
elif [ $WARNING_COUNT -gt 0 ]; then
    echo "⚠️  Clang-tidy found $WARNING_COUNT warnings"
    echo "This is informational - the build will not fail"
    exit 0
else
    echo "✅ No issues found!"
    exit 0
fi
