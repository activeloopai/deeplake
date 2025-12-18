#!/bin/bash

# Analyze timing statistics from the log file

if [ $# -eq 0 ]; then
    echo "Usage: $0 <timing_log_file>"
    echo "Example: $0 logs/timing_stats_20231218_143022.log"
    exit 1
fi

LOG_FILE="$1"

if [ ! -f "$LOG_FILE" ]; then
    echo "ERROR: Log file not found: $LOG_FILE"
    exit 1
fi

echo "==================================="
echo "Timing Analysis Report"
echo "==================================="
echo "Log file: $LOG_FILE"
echo ""

echo "=== Query Planning Times ==="
grep "Query Planning:" "$LOG_FILE" | awk '{print $4, $5}'

echo ""
echo "=== Direct Execution Plan Creation Times ==="
grep "Direct Execution Plan Creation:" "$LOG_FILE" | awk '{print $6, $7}'

echo ""
echo "=== Standard Planner Times ==="
grep "Standard Planner:" "$LOG_FILE" | awk '{print $4, $5}'

echo ""
echo "=== Executor Start Times ==="
grep "Executor Start:" "$LOG_FILE" | awk '{print $4, $5}'

echo ""
echo "=== Plan Analysis Times ==="
grep "Plan Analysis:" "$LOG_FILE" | awk '{print $4, $5}'

echo ""
echo "=== Executor Run Times ==="
grep "Executor Run:" "$LOG_FILE" | awk '{print $4, $5}'

echo ""
echo "=== Table Scan Begin Times ==="
grep "Table Scan Begin:" "$LOG_FILE" | awk '{print $5, $6}'

echo ""
echo "=== Table Scan End Times ==="
grep "Table Scan End:" "$LOG_FILE" | awk '{print $5, $6}'

echo ""
echo "=== DuckDB Query Execution Times ==="
grep "DuckDB query execution:" "$LOG_FILE" | awk '{print $5, $6}'

echo ""
echo "=== Fetching DuckDB Chunks Times ==="
grep "Fetching DuckDB chunks:" "$LOG_FILE" | awk '{print $5, $6}'

echo ""
echo "=== Executor End Times ==="
grep "Executor End:" "$LOG_FILE" | awk '{print $4, $5}'

echo ""
echo "=== Flush All Tables Times ==="
grep "Flush All Tables:" "$LOG_FILE" | awk '{print $5, $6}'

echo ""
echo "=== PostgreSQL Client Timing (Total) ==="
grep "Time:" "$LOG_FILE" | awk '{print $2, $3}'

echo ""
echo "==================================="
echo "Summary Statistics"
echo "==================================="

# Calculate averages
echo ""
echo "Average Query Planning Time:"
grep "Query Planning:" "$LOG_FILE" | awk '{sum+=$4; count++} END {if(count>0) print sum/count " ms"}'

echo "Average Executor Run Time:"
grep "Executor Run:" "$LOG_FILE" | awk '{sum+=$4; count++} END {if(count>0) print sum/count " ms"}'

echo "Average Flush All Tables Time:"
grep "Flush All Tables:" "$LOG_FILE" | awk '{sum+=$5; count++} END {if(count>0) print sum/count " ms"}'

echo "Average Total Time (from psql):"
grep "Time:" "$LOG_FILE" | awk '{gsub(/ms/, "", $2); sum+=$2; count++} END {if(count>0) print sum/count " ms"}'

echo ""
echo "==================================="
