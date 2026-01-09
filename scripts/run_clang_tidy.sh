#!/usr/bin/env bash

set -e

# Script to run clang-tidy on pg_deeplake source files
# Usage: ./scripts/run_clang_tidy.sh [build_dir]

trap "rm -rf ${TEMP_DIR}" EXIT

function run_tidy() {
  local output file_warnings file_errors
  output=$(clang-tidy --header-filter='.*deeplake_pg.*' -p "$BUILD_DIR" "$1" 2>&1 || true)
  file_warnings=$(echo "$output" | grep -c "deeplake_pg/.*warning:" || true)
  file_errors=$(echo "$output" | grep -c "deeplake_pg/.*error:" || true)
  echo "$file|$file_warnings|$file_errors" >"${TEMP_DIR}/${1}.count"
  if [ "$file_errors" -gt 0 ]; then
    echo "$output" | grep "deeplake_pg/.*error:" >"${TEMP_DIR}/${1}.output"
  elif [ "$file_warnings" -gt 0 ]; then
    echo "$output" | grep "deeplake_pg/.*warning:" >"${TEMP_DIR}/${1}.output"
  fi
}

function log() {
  local level ts
  level="$1"
  ts="$(date --utc -Iseconds)"
  shift
  printf "[%s] - [%s] - \"%s\"\n" "${level^^}" "${ts}" "$*"
}

SCRIPT_DIR="$(realpath "$(dirname "${BASH_SOURCE[0]}")")"
PROJECT_ROOT="$(cd $SCRIPT_DIR && realpath ..)"
BUILD_DIR="$(realpath "${1:-${PROJECT_ROOT}/builds/deeplake-pg-dev}")"

log info "running clang-tidy on cpp/deeplake_pg..."
log info "build directory: ${BUILD_DIR}"

if [ ! -f "${BUILD_DIR}/compile_commands.json" ]; then
  log error "compile_commands.json not found in ${BUILD_DIR}, please build the project first to generate compile_commands.json"
  exit 1
fi

cd "${PROJECT_ROOT}/cpp/deeplake_pg"
TEMP_DIR=$(mktemp -d)

log info "running clang-tidy in parallel..."

worker_count="$(nproc)"
for file in *.cpp; do
  timeout -k=1m run_tidy $file &
  while [ "$(jobs | wc -l)" -ge "$worker_count" ]; do
    sleep 0.1
  done
done
wait

log info "processing results..."

WARNINGS=0
ERRORS=0

for file in *.cpp; do
  if [ -f "${TEMP_DIR}/${file}.count" ]; then
    IFS='|' read -r fname FILE_WARNINGS FILE_ERRORS <"${TEMP_DIR}/${file}.count"

    if [ "$FILE_ERRORS" -gt 0 ]; then
      log error "❌ $file - has $FILE_ERRORS errors"
      cat "${TEMP_DIR}/${file}.output"
      ERRORS=$((ERRORS + FILE_ERRORS))
    elif [ "$FILE_WARNINGS" -gt 0 ]; then
      log warn "⚠ $file - has $FILE_WARNINGS warnings"
      cat "${TEMP_DIR}/${file}.output"
      WARNINGS=$((WARNINGS + FILE_WARNINGS))
    else
      log info "✓ $file - no issues"
    fi
  fi
done

log info "clang-tidy summary: warnings=$WARNINGS errors=$ERRORS"

if [ $ERRORS -gt 0 ]; then
  log error "❌ Clang-tidy found $ERRORS errors!"
  exit 1
elif [ $WARNINGS -gt 0 ]; then
  log warn "⚠ Clang-tidy found $WARNINGS warnings (non-blocking)"
  exit 0
else
  log info "✅ No issues found!"
  exit 0
fi

