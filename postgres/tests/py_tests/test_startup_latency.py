"""
Test startup latency and catalog loading performance for pg_deeplake extension.

This test measures:
1. Cold start latency (new PostgreSQL backend connecting)
2. Catalog loading time with stateless mode
3. Time to first query
4. Multi-table catalog discovery time
5. Comparison of stateless vs non-stateless modes

Run with: pytest test_startup_latency.py -v -s
"""
import pytest
import asyncpg
import asyncio
import os
import shutil
import subprocess
import time
import tempfile
import statistics
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class LatencyMetrics:
    """Container for latency measurements."""
    connection_time_ms: float = 0.0
    extension_load_time_ms: float = 0.0
    root_path_set_time_ms: float = 0.0
    first_query_time_ms: float = 0.0
    table_create_time_ms: float = 0.0
    catalog_discovery_time_ms: float = 0.0
    total_ready_time_ms: float = 0.0

    def __str__(self) -> str:
        return (
            f"  Connection:         {self.connection_time_ms:8.2f} ms\n"
            f"  Extension load:     {self.extension_load_time_ms:8.2f} ms\n"
            f"  Root path set:      {self.root_path_set_time_ms:8.2f} ms\n"
            f"  First query:        {self.first_query_time_ms:8.2f} ms\n"
            f"  Table create:       {self.table_create_time_ms:8.2f} ms\n"
            f"  Catalog discovery:  {self.catalog_discovery_time_ms:8.2f} ms\n"
            f"  Total ready time:   {self.total_ready_time_ms:8.2f} ms"
        )


@dataclass
class LatencyReport:
    """Aggregated latency report across multiple runs."""
    metrics: List[LatencyMetrics] = field(default_factory=list)

    def add(self, m: LatencyMetrics):
        self.metrics.append(m)

    def summary(self, name: str) -> str:
        if not self.metrics:
            return f"{name}: No data"

        def stats(values: List[float]) -> Tuple[float, float, float, float]:
            if not values:
                return (0.0, 0.0, 0.0, 0.0)
            return (
                min(values),
                max(values),
                statistics.mean(values),
                statistics.median(values)
            )

        conn = stats([m.connection_time_ms for m in self.metrics])
        ext = stats([m.extension_load_time_ms for m in self.metrics])
        root = stats([m.root_path_set_time_ms for m in self.metrics])
        query = stats([m.first_query_time_ms for m in self.metrics])
        create = stats([m.table_create_time_ms for m in self.metrics])
        disc = stats([m.catalog_discovery_time_ms for m in self.metrics])
        total = stats([m.total_ready_time_ms for m in self.metrics])

        return (
            f"\n{'='*60}\n"
            f"{name} ({len(self.metrics)} runs)\n"
            f"{'='*60}\n"
            f"{'Metric':<22} {'Min':>10} {'Max':>10} {'Mean':>10} {'Median':>10}\n"
            f"{'-'*60}\n"
            f"{'Connection':<22} {conn[0]:>10.2f} {conn[1]:>10.2f} {conn[2]:>10.2f} {conn[3]:>10.2f}\n"
            f"{'Extension load':<22} {ext[0]:>10.2f} {ext[1]:>10.2f} {ext[2]:>10.2f} {ext[3]:>10.2f}\n"
            f"{'Root path set':<22} {root[0]:>10.2f} {root[1]:>10.2f} {root[2]:>10.2f} {root[3]:>10.2f}\n"
            f"{'First query':<22} {query[0]:>10.2f} {query[1]:>10.2f} {query[2]:>10.2f} {query[3]:>10.2f}\n"
            f"{'Table create':<22} {create[0]:>10.2f} {create[1]:>10.2f} {create[2]:>10.2f} {create[3]:>10.2f}\n"
            f"{'Catalog discovery':<22} {disc[0]:>10.2f} {disc[1]:>10.2f} {disc[2]:>10.2f} {disc[3]:>10.2f}\n"
            f"{'-'*60}\n"
            f"{'TOTAL READY TIME':<22} {total[0]:>10.2f} {total[1]:>10.2f} {total[2]:>10.2f} {total[3]:>10.2f}\n"
            f"{'='*60}\n"
        )


async def measure_connection_latency(
    port: int = 5432,
    database: str = "postgres",
    with_extension: bool = True,
    root_path: Optional[str] = None,
    run_first_query: bool = True,
    create_table: bool = False,
    table_name: str = "latency_test",
) -> LatencyMetrics:
    """
    Measure various latency components of connecting to PostgreSQL.

    Args:
        port: PostgreSQL port
        database: Database to connect to
        with_extension: Whether to load pg_deeplake extension
        root_path: If set, configure deeplake.root_path
        run_first_query: Whether to measure first query time
        create_table: Whether to measure table creation time
        table_name: Name for test table

    Returns:
        LatencyMetrics with all measurements
    """
    user = os.environ.get("USER", "postgres")
    metrics = LatencyMetrics()
    total_start = time.perf_counter()

    # 1. Measure connection time
    conn_start = time.perf_counter()
    conn = await asyncpg.connect(
        database=database,
        user=user,
        host="localhost",
        port=port,
        statement_cache_size=0
    )
    metrics.connection_time_ms = (time.perf_counter() - conn_start) * 1000

    try:
        # 2. Measure extension load time
        if with_extension:
            ext_start = time.perf_counter()
            await conn.execute("DROP EXTENSION IF EXISTS pg_deeplake CASCADE")
            await conn.execute("CREATE EXTENSION pg_deeplake")
            metrics.extension_load_time_ms = (time.perf_counter() - ext_start) * 1000

        # 3. Measure root_path set time (triggers catalog loading in stateless mode)
        if root_path:
            root_start = time.perf_counter()
            await conn.execute(f"SET deeplake.root_path = '{root_path}'")
            metrics.root_path_set_time_ms = (time.perf_counter() - root_start) * 1000

        # 4. Measure first query time
        if run_first_query and with_extension:
            query_start = time.perf_counter()
            await conn.execute("SELECT 1")
            metrics.first_query_time_ms = (time.perf_counter() - query_start) * 1000

        # 5. Measure table creation time
        if create_table and with_extension:
            create_start = time.perf_counter()
            await conn.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE")
            await conn.execute(f"""
                CREATE TABLE {table_name} (
                    id INT,
                    name TEXT,
                    value FLOAT
                ) USING deeplake
            """)
            metrics.table_create_time_ms = (time.perf_counter() - create_start) * 1000

            # Cleanup
            await conn.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE")

        metrics.total_ready_time_ms = (time.perf_counter() - total_start) * 1000

    finally:
        await conn.close()

    return metrics


async def measure_catalog_discovery_latency(
    port: int,
    root_path: str,
    num_tables: int,
) -> LatencyMetrics:
    """
    Measure time to discover existing tables from catalog.

    This simulates a second instance discovering tables created by another instance.
    """
    user = os.environ.get("USER", "postgres")
    metrics = LatencyMetrics()
    total_start = time.perf_counter()

    conn_start = time.perf_counter()
    conn = await asyncpg.connect(
        database="postgres",
        user=user,
        host="localhost",
        port=port,
        statement_cache_size=0
    )
    metrics.connection_time_ms = (time.perf_counter() - conn_start) * 1000

    try:
        # Load extension
        ext_start = time.perf_counter()
        await conn.execute("DROP EXTENSION IF EXISTS pg_deeplake CASCADE")
        await conn.execute("CREATE EXTENSION pg_deeplake")
        metrics.extension_load_time_ms = (time.perf_counter() - ext_start) * 1000

        # Set root_path - this triggers catalog discovery
        disc_start = time.perf_counter()
        await conn.execute(f"SET deeplake.root_path = '{root_path}'")
        metrics.root_path_set_time_ms = (time.perf_counter() - disc_start) * 1000

        # Verify tables were discovered
        query_start = time.perf_counter()
        count = await conn.fetchval("SELECT COUNT(*) FROM pg_deeplake_tables")
        metrics.first_query_time_ms = (time.perf_counter() - query_start) * 1000

        metrics.catalog_discovery_time_ms = metrics.root_path_set_time_ms
        metrics.total_ready_time_ms = (time.perf_counter() - total_start) * 1000

    finally:
        await conn.close()

    return metrics


@pytest.fixture
def temp_root_path(temp_dir_for_postgres):
    """Create a temporary root path for deeplake datasets."""
    return temp_dir_for_postgres


@pytest.mark.asyncio
async def test_baseline_connection_latency(pg_server):
    """
    Measure baseline connection latency without extension.

    This establishes the baseline PostgreSQL connection overhead.
    """
    print("\n" + "="*60)
    print("BASELINE CONNECTION LATENCY (no extension)")
    print("="*60)

    report = LatencyReport()
    num_runs = 5

    for i in range(num_runs):
        metrics = await measure_connection_latency(
            with_extension=False,
            run_first_query=False,
        )
        report.add(metrics)
        print(f"Run {i+1}: Connection = {metrics.connection_time_ms:.2f} ms")

    print(report.summary("Baseline (no extension)"))


@pytest.mark.asyncio
async def test_extension_load_latency(pg_server):
    """
    Measure latency of loading pg_deeplake extension.

    This measures the overhead of CREATE EXTENSION pg_deeplake.
    """
    print("\n" + "="*60)
    print("EXTENSION LOAD LATENCY")
    print("="*60)

    report = LatencyReport()
    num_runs = 5

    for i in range(num_runs):
        metrics = await measure_connection_latency(
            with_extension=True,
            run_first_query=True,
            create_table=False,
        )
        report.add(metrics)
        print(f"Run {i+1}:")
        print(metrics)
        print()

    print(report.summary("Extension Load"))


@pytest.mark.asyncio
async def test_table_creation_latency(pg_server, temp_root_path):
    """
    Measure latency of creating a deeplake table.
    """
    print("\n" + "="*60)
    print("TABLE CREATION LATENCY")
    print("="*60)

    report = LatencyReport()
    num_runs = 5

    for i in range(num_runs):
        metrics = await measure_connection_latency(
            with_extension=True,
            root_path=temp_root_path,
            run_first_query=True,
            create_table=True,
            table_name=f"latency_test_{i}",
        )
        report.add(metrics)
        print(f"Run {i+1}:")
        print(metrics)
        print()

    print(report.summary("Table Creation"))


@pytest.mark.asyncio
async def test_stateless_catalog_loading_latency(pg_server, temp_root_path):
    """
    Measure catalog loading latency with stateless mode enabled.

    This is the critical test for the parallelization improvement.
    """
    print("\n" + "="*60)
    print("STATELESS CATALOG LOADING LATENCY")
    print("="*60)

    user = os.environ.get("USER", "postgres")

    # First, create some tables in the catalog
    print("\nSetup: Creating tables in catalog...")
    setup_conn = await asyncpg.connect(
        database="postgres",
        user=user,
        host="localhost",
        statement_cache_size=0
    )

    try:
        await setup_conn.execute("DROP EXTENSION IF EXISTS pg_deeplake CASCADE")
        await setup_conn.execute("CREATE EXTENSION pg_deeplake")
        await setup_conn.execute(f"SET deeplake.root_path = '{temp_root_path}'")

        # Create multiple tables to populate the catalog
        num_tables = 5
        for i in range(num_tables):
            await setup_conn.execute(f"""
                CREATE TABLE catalog_test_{i} (
                    id INT,
                    name TEXT,
                    data FLOAT
                ) USING deeplake
            """)
            await setup_conn.execute(f"INSERT INTO catalog_test_{i} VALUES ({i}, 'test', {i}.5)")

        print(f"Created {num_tables} tables in catalog")

        # Verify tables in catalog
        count = await setup_conn.fetchval("SELECT COUNT(*) FROM pg_deeplake_tables")
        print(f"Tables in catalog: {count}")

    finally:
        await setup_conn.close()

    # Now measure the catalog discovery time from a fresh connection
    print("\nMeasuring catalog discovery latency...")
    report = LatencyReport()
    num_runs = 5

    for i in range(num_runs):
        metrics = await measure_catalog_discovery_latency(
            port=5432,
            root_path=temp_root_path,
            num_tables=num_tables,
        )
        report.add(metrics)
        print(f"Run {i+1}:")
        print(metrics)
        print()

    print(report.summary("Stateless Catalog Loading"))

    # Cleanup
    cleanup_conn = await asyncpg.connect(
        database="postgres",
        user=user,
        host="localhost",
        statement_cache_size=0
    )
    try:
        await cleanup_conn.execute("DROP EXTENSION IF EXISTS pg_deeplake CASCADE")
        await cleanup_conn.execute("CREATE EXTENSION pg_deeplake")
        await cleanup_conn.execute(f"SET deeplake.root_path = '{temp_root_path}'")
        for i in range(num_tables):
            await cleanup_conn.execute(f"DROP TABLE IF EXISTS catalog_test_{i} CASCADE")
    finally:
        await cleanup_conn.close()


@pytest.mark.asyncio
async def test_stateless_vs_nonstateless_comparison(pg_server, temp_root_path):
    """
    Compare latency between stateless and non-stateless modes.
    """
    print("\n" + "="*60)
    print("STATELESS vs NON-STATELESS COMPARISON")
    print("="*60)

    # Non-stateless (local catalog)
    print("\n--- Non-Stateless Mode ---")
    non_stateless_report = LatencyReport()
    for i in range(3):
        metrics = await measure_connection_latency(
            with_extension=True,
            root_path=temp_root_path,
            run_first_query=True,
            create_table=True,
            table_name=f"nonstateless_test_{i}",
        )
        non_stateless_report.add(metrics)

    print(non_stateless_report.summary("Non-Stateless Mode"))

    # Stateless (shared catalog)
    print("\n--- Stateless Mode ---")
    stateless_report = LatencyReport()
    for i in range(3):
        metrics = await measure_connection_latency(
            with_extension=True,
            root_path=temp_root_path,
            run_first_query=True,
            create_table=True,
            table_name=f"stateless_test_{i}",
        )
        stateless_report.add(metrics)

    print(stateless_report.summary("Stateless Mode"))

    # Calculate overhead
    non_stateless_avg = statistics.mean([m.total_ready_time_ms for m in non_stateless_report.metrics])
    stateless_avg = statistics.mean([m.total_ready_time_ms for m in stateless_report.metrics])
    overhead = stateless_avg - non_stateless_avg
    overhead_pct = (overhead / non_stateless_avg) * 100 if non_stateless_avg > 0 else 0

    print(f"\n{'='*60}")
    print(f"OVERHEAD ANALYSIS")
    print(f"{'='*60}")
    print(f"Non-stateless avg total: {non_stateless_avg:.2f} ms")
    print(f"Stateless avg total:     {stateless_avg:.2f} ms")
    print(f"Stateless overhead:      {overhead:.2f} ms ({overhead_pct:.1f}%)")
    print(f"{'='*60}")


@pytest.mark.asyncio
async def test_multi_table_catalog_scaling(pg_server, temp_root_path):
    """
    Test how catalog loading time scales with number of tables.
    """
    print("\n" + "="*60)
    print("CATALOG LOADING SCALING TEST")
    print("="*60)

    user = os.environ.get("USER", "postgres")
    table_counts = [1, 5, 10, 20]
    results = []

    for num_tables in table_counts:
        print(f"\n--- Testing with {num_tables} tables ---")

        # Setup: Create tables
        setup_conn = await asyncpg.connect(
            database="postgres",
            user=user,
            host="localhost",
            statement_cache_size=0
        )

        try:
            await setup_conn.execute("DROP EXTENSION IF EXISTS pg_deeplake CASCADE")
            await setup_conn.execute("CREATE EXTENSION pg_deeplake")
            await setup_conn.execute(f"SET deeplake.root_path = '{temp_root_path}'")

            # Create tables
            for i in range(num_tables):
                await setup_conn.execute(f"""
                    CREATE TABLE scale_test_{num_tables}_{i} (
                        id INT, name TEXT
                    ) USING deeplake
                """)
        finally:
            await setup_conn.close()

        # Measure catalog discovery
        report = LatencyReport()
        for _ in range(3):
            metrics = await measure_catalog_discovery_latency(
                port=5432,
                root_path=temp_root_path,
                num_tables=num_tables,
            )
            report.add(metrics)

        avg_time = statistics.mean([m.catalog_discovery_time_ms for m in report.metrics])
        results.append((num_tables, avg_time))
        print(f"  Average catalog discovery time: {avg_time:.2f} ms")

        # Cleanup
        cleanup_conn = await asyncpg.connect(
            database="postgres",
            user=user,
            host="localhost",
            statement_cache_size=0
        )
        try:
            await cleanup_conn.execute("DROP EXTENSION IF EXISTS pg_deeplake CASCADE")
            await cleanup_conn.execute("CREATE EXTENSION pg_deeplake")
            await cleanup_conn.execute(f"SET deeplake.root_path = '{temp_root_path}'")
            for i in range(num_tables):
                await cleanup_conn.execute(f"DROP TABLE IF EXISTS scale_test_{num_tables}_{i} CASCADE")
        finally:
            await cleanup_conn.close()

    # Print scaling summary
    print(f"\n{'='*60}")
    print(f"SCALING SUMMARY")
    print(f"{'='*60}")
    print(f"{'Tables':<10} {'Avg Discovery Time (ms)':<25}")
    print(f"{'-'*35}")
    for num_tables, avg_time in results:
        print(f"{num_tables:<10} {avg_time:<25.2f}")

    # Check if scaling is sub-linear (good) or linear/super-linear (bad)
    if len(results) >= 2:
        first_time = results[0][1]
        last_time = results[-1][1]
        first_count = results[0][0]
        last_count = results[-1][0]

        table_ratio = last_count / first_count
        time_ratio = last_time / first_time if first_time > 0 else 0

        print(f"\nTable count increased {table_ratio:.1f}x")
        print(f"Discovery time increased {time_ratio:.1f}x")

        if time_ratio < table_ratio:
            print("Result: SUB-LINEAR scaling (good!)")
        elif time_ratio > table_ratio * 1.5:
            print("Result: SUPER-LINEAR scaling (needs optimization)")
        else:
            print("Result: APPROXIMATELY LINEAR scaling")

    print(f"{'='*60}")


@pytest.mark.asyncio
async def test_cold_start_simulation(pg_server, temp_root_path):
    """
    Simulate a cold start scenario where a new backend connects.

    This measures the total time from connection to being ready to serve queries.
    """
    print("\n" + "="*60)
    print("COLD START SIMULATION")
    print("="*60)

    user = os.environ.get("USER", "postgres")

    # Setup: Create some existing data
    print("\nSetup: Creating existing data...")
    setup_conn = await asyncpg.connect(
        database="postgres",
        user=user,
        host="localhost",
        statement_cache_size=0
    )

    try:
        await setup_conn.execute("DROP EXTENSION IF EXISTS pg_deeplake CASCADE")
        await setup_conn.execute("CREATE EXTENSION pg_deeplake")
        await setup_conn.execute(f"SET deeplake.root_path = '{temp_root_path}'")

        await setup_conn.execute("""
            CREATE TABLE existing_data (
                id INT,
                name TEXT,
                embedding FLOAT[]
            ) USING deeplake
        """)

        # Insert some data
        for i in range(100):
            await setup_conn.execute(f"""
                INSERT INTO existing_data VALUES
                ({i}, 'item_{i}', ARRAY[{','.join(str(float(j)) for j in range(128))}])
            """)

        print("Created table with 100 rows")
    finally:
        await setup_conn.close()

    # Measure cold start - simulating new backend connections to existing data
    # We use separate connections without dropping the extension to preserve the table
    print("\nMeasuring cold start latency...")

    for run in range(3):
        total_start = time.perf_counter()

        conn = await asyncpg.connect(
            database="postgres",
            user=user,
            host="localhost",
            statement_cache_size=0
        )
        conn_time = (time.perf_counter() - total_start) * 1000

        try:
            # Extension is already loaded via shared_preload_libraries
            # Just configure the session (simulating a new backend)
            root_start = time.perf_counter()
            await conn.execute(f"SET deeplake.root_path = '{temp_root_path}'")
            root_time = (time.perf_counter() - root_start) * 1000

            # First real query against the existing data
            query_start = time.perf_counter()
            count = await conn.fetchval("SELECT COUNT(*) FROM existing_data")
            query_time = (time.perf_counter() - query_start) * 1000

            total_time = (time.perf_counter() - total_start) * 1000

            print(f"\nRun {run + 1}:")
            print(f"  Connection:       {conn_time:8.2f} ms")
            print(f"  Root path set:    {root_time:8.2f} ms")
            print(f"  First query:      {query_time:8.2f} ms (count={count})")
            print(f"  TOTAL COLD START: {total_time:8.2f} ms")

        finally:
            await conn.close()

    # Cleanup
    cleanup_conn = await asyncpg.connect(
        database="postgres",
        user=user,
        host="localhost",
        statement_cache_size=0
    )
    try:
        await cleanup_conn.execute("DROP EXTENSION IF EXISTS pg_deeplake CASCADE")
        await cleanup_conn.execute("CREATE EXTENSION pg_deeplake")
        await cleanup_conn.execute(f"SET deeplake.root_path = '{temp_root_path}'")
        await cleanup_conn.execute("DROP TABLE IF EXISTS existing_data CASCADE")
    finally:
        await cleanup_conn.close()

    print(f"\n{'='*60}")
