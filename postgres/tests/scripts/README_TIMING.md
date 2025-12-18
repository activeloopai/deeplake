# Timing Measurement Scripts

This directory contains scripts for measuring and analyzing query execution timing in the pg_deeplake PostgreSQL extension.

## Quick Start

### 1. Build Extension and Run Full Test

```bash
./build_and_time_insert.sh
```

This will build the extension and run a comprehensive insert test with timing measurements.

### 2. Quick Test (Verify Timing Works)

```bash
./test_timing_simple.sh
```

This runs a simple test to verify that timing measurements are enabled and working.

### 3. Analyze Timing Results

```bash
./analyze_timing.sh logs/timing_stats_<timestamp>.log
```

This provides detailed analysis of a timing log file.

## Available Scripts

### `build_and_time_insert.sh`
**Purpose**: Build the extension and run insert timing test
**Usage**: `./build_and_time_insert.sh`
**What it does**:
- Builds the pg_deeplake extension
- Runs the insert test with timing enabled
- Saves timing statistics to a log file
- Displays a timing summary

### `time_insert_clean.sh`
**Purpose**: Run insert test with timing (extension must already be built)
**Usage**: `./time_insert_clean.sh`
**What it does**:
- Starts PostgreSQL
- Sets up the test schema
- Runs INSERT queries with timing enabled
- Saves timing statistics to a log file
- Displays a timing summary

### `test_timing_simple.sh`
**Purpose**: Quick test to verify timing measurements work
**Usage**: `./test_timing_simple.sh`
**Prerequisites**: PostgreSQL must be running
**What it does**:
- Runs simple queries with timing enabled
- Shows current settings
- Verifies timing output is visible

### `analyze_timing.sh`
**Purpose**: Analyze timing statistics from a log file
**Usage**: `./analyze_timing.sh <log_file>`
**Example**: `./analyze_timing.sh logs/timing_stats_20231218_143022.log`
**What it does**:
- Extracts all timing measurements
- Calculates averages
- Displays summary statistics

## Workflow

### Standard Workflow

```bash
# 1. Build and run test
./build_and_time_insert.sh

# 2. Analyze the results
./analyze_timing.sh logs/timing_stats_<timestamp>.log
```

### Quick Verification

```bash
# Just verify timing is working
./test_timing_simple.sh
```

### Run Test Without Rebuild

```bash
# If you've already built the extension
./time_insert_clean.sh
```

## Output Files

All timing logs are saved to: `../logs/timing_stats_<timestamp>.log`

Example filename: `logs/timing_stats_20231218_143022.log`

## What Gets Measured

The timing instrumentation covers:

1. **Query Planning**: Time to plan the query
2. **Executor Start**: Initialization time
3. **Executor Run**: Main execution time
4. **Table Scan Operations**: Time to begin/end table scans
5. **DuckDB Execution**: Time for DuckDB query execution (if using direct executor)
6. **Flush Operations**: Time to flush data to storage (for INSERT/UPDATE/DELETE)
7. **Executor End**: Cleanup time

## Example Output

```
=== Timing Summary ===
INFO:  Query Planning: 2.35 ms
INFO:  Executor Start: 1.20 ms
INFO:  Plan Analysis: 0.80 ms
INFO:  Executor Run: 150.45 ms
INFO:  Table Scan Begin: 5.10 ms
INFO:  Table Scan End: 0.50 ms
INFO:  Flush All Tables: 45.30 ms
INFO:  Executor End: 2.10 ms
Time: 207.850 ms
```

## Configuration

Timing measurements are controlled by the GUC parameter:

```sql
-- Enable timing (default: false)
SET pg_deeplake.print_runtime_stats = true;

-- Check current setting
SHOW pg_deeplake.print_runtime_stats;
```

## Troubleshooting

### No timing output?

1. Make sure you rebuilt the extension after adding timing code
2. Check that `pg_deeplake.print_runtime_stats = true`
3. Check that `client_min_messages = INFO` or lower
4. Make sure PostgreSQL was restarted after rebuild

### Only seeing some timing messages?

Some measurements only appear for specific query types:
- **Flush All Tables**: Only for INSERT/UPDATE/DELETE
- **Direct Execution Plan**: Only when DeepLake executor is used
- **Standard Planner**: Only as fallback when direct executor can't handle query

## Advanced Usage

### Compare Execution Paths

```bash
# Test with DeepLake executor
psql -d postgres -c "SET pg_deeplake.use_deeplake_executor = true; \
                     SET pg_deeplake.print_runtime_stats = true; \
                     SELECT * FROM lineitem LIMIT 1000;"

# Test with standard PostgreSQL path
psql -d postgres -c "SET pg_deeplake.use_deeplake_executor = false; \
                     SET pg_deeplake.print_runtime_stats = true; \
                     SELECT * FROM lineitem LIMIT 1000;"
```

### Extract Specific Metrics

```bash
# Show only DuckDB execution times
grep "DuckDB" logs/timing_stats_*.log

# Show only flush times
grep "Flush All Tables" logs/timing_stats_*.log

# Show planning times
grep "Query Planning" logs/timing_stats_*.log
```

## Performance Impact

- Timing measurements use `std::chrono::high_resolution_clock`
- Overhead: ~1-2 microseconds per measurement
- Only active when `print_runtime_stats = true`
- Zero impact when disabled (default)

## Documentation

See [TIMING_MEASUREMENTS.md](TIMING_MEASUREMENTS.md) for comprehensive documentation including:
- Detailed explanation of all timing phases
- Code locations for each measurement
- Understanding the timing hierarchy
- Troubleshooting guide
- Examples and use cases

## Directory Structure

```
scripts/
├── build_and_time_insert.sh       # Build and run full test
├── time_insert_clean.sh            # Run test only
├── test_timing_simple.sh           # Quick verification test
├── analyze_timing.sh               # Analyze timing logs
├── TIMING_MEASUREMENTS.md          # Comprehensive documentation
└── README_TIMING.md               # This file
```

## Support

For questions or issues:
1. Check [TIMING_MEASUREMENTS.md](TIMING_MEASUREMENTS.md) for detailed documentation
2. Review the timing log files in `logs/`
3. Run `test_timing_simple.sh` to verify basic functionality
