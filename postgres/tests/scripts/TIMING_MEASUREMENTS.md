# Query Execution Timing Measurements

This document describes how to measure and analyze query execution timing in the pg_deeplake PostgreSQL extension.

## Overview

The extension includes comprehensive timing instrumentation at all key execution points:

1. **Query Planning Phase**
   - Overall query planning
   - Direct execution plan creation (DeepLake executor)
   - Standard PostgreSQL planner fallback

2. **Executor Hooks**
   - Executor start (initialization)
   - Plan analysis
   - Executor run (main execution)
   - Executor end (cleanup)
   - Flush all tables (for INSERT/UPDATE/DELETE)

3. **Table Scan Operations**
   - Table scan begin (setup)
   - Table scan end (cleanup)

4. **DeepLake Custom Executor**
   - Begin scan
   - End scan

5. **DuckDB Query Execution**
   - Query execution
   - Chunk fetching

## Quick Start

### Option 1: Build and Run Test (Recommended)

```bash
cd /home/azureuser/work/deeplake/postgres/tests/scripts
./build_and_time_insert.sh
```

This will:
1. Build the PostgreSQL extension with your timing changes
2. Run the insert test with timing measurements enabled
3. Save timing statistics to a log file
4. Display a timing summary

### Option 2: Run Test Only (If Already Built)

```bash
cd /home/azureuser/work/deeplake/postgres/tests/scripts
./time_insert_clean.sh
```

## Analyzing Timing Results

### View Summary During Test

The test script automatically displays a summary at the end showing:
- Query Planning times
- Executor times
- Table Scan times
- DuckDB execution times
- Flush times
- Total PostgreSQL client times

### Detailed Analysis

Use the analysis script on any timing log file:

```bash
./analyze_timing.sh logs/timing_stats_20231218_143022.log
```

This will show:
- Individual timing measurements for each phase
- Average times across multiple runs
- Summary statistics

### Manual Analysis

View the full log file directly:

```bash
less logs/timing_stats_<timestamp>.log
```

## Enabling Timing Measurements

### In SQL Session

```sql
-- Enable timing measurements
SET pg_deeplake.print_runtime_stats = true;

-- Set log level to see INFO messages
SET client_min_messages = INFO;

-- Enable psql timing
\timing on

-- Run your query
SELECT * FROM your_table;
```

### In postgresql.conf

Add to `postgresql.conf`:

```
pg_deeplake.print_runtime_stats = true
```

Then restart PostgreSQL.

## Understanding the Output

### Timing Message Format

All timing messages follow this format:
```
INFO:  <Phase Name>: <duration> ms
```

Example:
```
INFO:  Query Planning: 2.35 ms
INFO:  Executor Start: 1.20 ms
INFO:  Table Scan Begin: 5.10 ms
INFO:  Executor Run: 150.45 ms
INFO:  Table Scan End: 0.50 ms
INFO:  Flush All Tables: 45.30 ms
INFO:  Executor End: 2.10 ms
```

### Timing Hierarchy

```
Query Planning
├── Direct Execution Plan Creation (if using DeepLake executor)
└── Standard Planner (fallback)

Executor Start
└── Plan Analysis (if needed)

Executor Run
├── Table Scan Begin
│   └── (per table)
├── [Query execution happens here]
└── Table Scan End
    └── (per table)

Executor End
└── Flush All Tables (for INSERT/UPDATE/DELETE)
```

### For DeepLake Direct Executor

```
BeginScan DeeplakeExecutor
├── DuckDB query execution
└── Fetching DuckDB chunks

[Tuple fetching happens during Executor Run]

EndScan DeeplakeExecutor
```

## Code Locations

The timing instrumentation is added in these files:

- `cpp/deeplake_pg/extension_init.cpp`: Planner and executor hooks
- `cpp/deeplake_pg/table_am.cpp`: Table scan operations
- `cpp/deeplake_pg/deeplake_executor.cpp`: DeepLake custom executor
- `cpp/deeplake_pg/duckdb_executor.cpp`: DuckDB query execution

The timing infrastructure is in:
- `cpp/deeplake_pg/reporter.hpp`: `runtime_printer` class

## Troubleshooting

### No Timing Output

If you don't see timing messages:

1. **Check if print_runtime_stats is enabled:**
   ```sql
   SHOW pg_deeplake.print_runtime_stats;
   ```

2. **Check log level:**
   ```sql
   SHOW client_min_messages;
   ```
   Should be `INFO` or lower.

3. **Rebuild the extension:**
   ```bash
   cd /home/azureuser/work/deeplake
   python3 scripts/build_pg_ext.py --major-version 18 --minor-version 0
   ```

### Partial Timing Output

Some phases may not show timing if they're not executed:
- **Direct Execution Plan Creation**: Only shown if `pg_deeplake.use_deeplake_executor = true`
- **Standard Planner**: Only shown as fallback when direct executor can't handle the query
- **Flush All Tables**: Only shown for INSERT/UPDATE/DELETE operations
- **Plan Analysis**: Only shown for certain query types

## Performance Impact

The timing measurements use `std::chrono::high_resolution_clock` and have minimal overhead:
- ~1-2 microseconds per measurement
- Only active when `pg_deeplake.print_runtime_stats = true`
- No impact when disabled (default)

## Examples

### Example 1: INSERT Query Timing

```
INFO:  Query Planning: 1.23 ms
INFO:  Executor Start: 0.85 ms
INFO:  Executor Run: 125.40 ms
INFO:  Table Scan Begin: 0.15 ms
INFO:  Table Scan End: 0.12 ms
INFO:  Flush All Tables: 42.50 ms
INFO:  Executor End: 1.05 ms
Time: 171.234 ms
```

### Example 2: SELECT Query with DeepLake Executor

```
INFO:  Query Planning: 2.15 ms
INFO:  Direct Execution Plan Creation: 1.80 ms
INFO:  Executor Start: 0.95 ms
INFO:  BeginScan DeeplakeExecutor: 8.50 ms
INFO:  DuckDB query execution: 45.20 ms
INFO:  Fetching DuckDB chunks: 12.30 ms
INFO:  Executor Run: 68.75 ms
INFO:  EndScan DeeplakeExecutor: 0.45 ms
INFO:  Executor End: 0.85 ms
Time: 82.150 ms
```

## Advanced Usage

### Compare Different Approaches

Run the same query with different settings to compare:

```sql
-- Test 1: With DeepLake executor
SET pg_deeplake.use_deeplake_executor = true;
SET pg_deeplake.print_runtime_stats = true;
\timing on
SELECT * FROM lineitem LIMIT 1000;

-- Test 2: Without DeepLake executor (standard path)
SET pg_deeplake.use_deeplake_executor = false;
SELECT * FROM lineitem LIMIT 1000;
```

### Profile Specific Operations

Focus on specific phases by grepping the log:

```bash
# Focus on DuckDB execution
grep "DuckDB" logs/timing_stats_*.log

# Focus on flush operations
grep "Flush" logs/timing_stats_*.log

# Focus on planning
grep "Planning\|Planner" logs/timing_stats_*.log
```

## Notes

- All times are in milliseconds (ms) unless otherwise specified
- Times represent wall-clock time (not CPU time)
- Nested timers are included in parent timers
- The PostgreSQL client `\timing` shows total time including network overhead
