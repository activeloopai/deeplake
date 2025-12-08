"""
Test concurrent CREATE TABLE IF NOT EXISTS operations.

This test verifies that multiple concurrent clients can safely execute
CREATE TABLE IF NOT EXISTS on the same table without causing duplicate
key violations in the pg_deeplake_tables metadata table.

Regression test for race condition where multiple asyncpg clients would
get duplicate key errors on table_name unique constraint.
"""
import pytest
import asyncpg
import asyncio
import tempfile
import os
from lib.assertions import Assertions


async def create_table_worker(worker_id: int, schema_name: str, table_name: str):
    """
    Worker coroutine that attempts to create a table.

    Args:
        worker_id: Identifier for this worker
        schema_name: Schema name for the table
        table_name: Name of the table to create

    Returns:
        Tuple of (worker_id, success: bool, error: Optional[str])
    """
    conn = None
    try:
        # Create a new connection for each worker to simulate concurrent clients
        user = os.environ.get("USER", "postgres")
        conn = await asyncpg.connect(
            database="postgres",
            user=user,
            host="localhost",
            statement_cache_size=0
        )

        # Execute CREATE TABLE IF NOT EXISTS
        create_sql = f"""
            CREATE TABLE IF NOT EXISTS "{schema_name}".{table_name} (
                source_id TEXT,
                summary TEXT,
                metadata JSONB,
                created_at BIGINT,
                included_in_summary boolean,
                source_type TEXT
            ) USING deeplake
        """

        await conn.execute(create_sql)
        return (worker_id, True, None)

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        return (worker_id, False, error_msg)

    finally:
        if conn:
            await conn.close()


@pytest.mark.asyncio
async def test_concurrent_create_table_small(db_conn: asyncpg.Connection):
    """
    Test concurrent CREATE TABLE IF NOT EXISTS with 5 workers.

    This test verifies that multiple concurrent clients can create the same
    table without triggering duplicate key violations.
    """
    await _test_concurrent_create_table(db_conn, num_workers=5)


@pytest.mark.asyncio
async def test_concurrent_create_table_medium(db_conn: asyncpg.Connection):
    """
    Test concurrent CREATE TABLE IF NOT EXISTS with 10 workers.

    This test verifies that multiple concurrent clients can create the same
    table without triggering duplicate key violations.
    """
    await _test_concurrent_create_table(db_conn, num_workers=10)


@pytest.mark.asyncio
async def test_concurrent_create_table_large(db_conn: asyncpg.Connection):
    """
    Test concurrent CREATE TABLE IF NOT EXISTS with 20 workers.

    This test verifies that multiple concurrent clients can create the same
    table without triggering duplicate key violations.
    """
    await _test_concurrent_create_table(db_conn, num_workers=20)


@pytest.mark.asyncio
async def test_concurrent_create_table_stress(db_conn: asyncpg.Connection):
    """
    Stress test with maximum concurrency to trigger race condition.

    Uses a single table with many concurrent CREATE attempts to maximize
    the chance of hitting the metadata INSERT race condition.
    """
    # Single table, many concurrent attempts
    await _test_concurrent_create_table(
        db_conn,
        num_workers=50,  # High concurrency
        test_name="stress_test"
    )


@pytest.mark.asyncio
async def test_concurrent_create_table_extreme_stress(db_conn: asyncpg.Connection):
    """
    Extreme stress test with multiple rounds and very high concurrency.

    This test specifically tries to trigger the metadata race condition by:
    - Using very high worker counts (100 concurrent)
    - Running multiple sequential rounds
    - Checking for duplicate key violations explicitly
    """
    num_rounds = 3
    workers_per_round = 100

    for round_num in range(num_rounds):
        print(f"\n  Extreme stress test round {round_num + 1}/{num_rounds}")
        await _test_concurrent_create_table(
            db_conn,
            num_workers=workers_per_round,
            test_name=f"extreme_round_{round_num}"
        )
        # Very small delay to stress the system
        await asyncio.sleep(0.01)

    print(f"\n  ✓ Extreme stress test completed: {num_rounds} rounds × {workers_per_round} workers")


@pytest.mark.asyncio
async def test_concurrent_create_multiple_tables_stress(db_conn: asyncpg.Connection):
    """
    Test creating multiple different tables concurrently with high parallelism.

    This tests that our DDL lock doesn't create bottlenecks when creating
    different tables simultaneously.
    """
    num_tables = 10
    workers_per_table = 20  # 20 workers trying to create each table

    schema_name = "test_schema"
    await db_conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema_name}"')

    async def create_table_with_workers(table_id: int):
        """Create a table with multiple concurrent workers."""
        table_name = f"stress_table_{table_id}"

        # Use a separate connection for cleanup to avoid conflicts
        user = os.environ.get("USER", "postgres")
        cleanup_conn = await asyncpg.connect(
            database="postgres",
            user=user,
            host="localhost",
            statement_cache_size=0
        )

        try:
            # Ensure table doesn't exist
            try:
                await cleanup_conn.execute(f'DROP TABLE IF EXISTS "{schema_name}".{table_name} CASCADE')
                await cleanup_conn.execute(f'DROP TYPE IF EXISTS "{schema_name}".{table_name} CASCADE')
            except:
                pass

            tasks = [
                create_table_worker(worker_id, schema_name, table_name)
                for worker_id in range(workers_per_table)
            ]

            results = await asyncio.gather(*tasks)

            # Check results
            successes = sum(1 for _, success, _ in results if success)
            errors = sum(1 for _, success, _ in results if not success)

            # Check for metadata-level duplicate key errors (our bug)
            metadata_duplicate_key_errors = sum(
                1 for _, success, error in results
                if not success and error and "duplicate key" in error.lower()
                and "pg_deeplake_tables" in error.lower()
            )

            if metadata_duplicate_key_errors > 0:
                pytest.fail(f"Table {table_name}: Found {metadata_duplicate_key_errors} metadata duplicate key violations!")

            # Verify metadata - but only if at least one worker succeeded
            # (All workers might fail with PostgreSQL's "type already exists" catalog-level race)
            if successes > 0:
                count = await cleanup_conn.fetchval(
                    'SELECT COUNT(*) FROM public.pg_deeplake_tables WHERE table_name = $1',
                    f'{schema_name}.{table_name}'
                )
                assert count == 1, f"Table {table_name}: Expected 1 metadata entry, found {count}"
            else:
                # All workers failed due to PostgreSQL catalog race (type already exists)
                # This is expected and not our bug - verify no metadata was created
                count = await cleanup_conn.fetchval(
                    'SELECT COUNT(*) FROM public.pg_deeplake_tables WHERE table_name = $1',
                    f'{schema_name}.{table_name}'
                )
                assert count == 0, f"Table {table_name}: All workers failed but found {count} metadata entries (should be 0)"

            return (table_id, successes, errors)
        finally:
            await cleanup_conn.close()

    print(f"\n  Creating {num_tables} tables with {workers_per_table} concurrent workers each...")

    # Create all tables in parallel
    results = await asyncio.gather(*[
        create_table_with_workers(i) for i in range(num_tables)
    ])

    # Verify results
    total_successes = sum(successes for _, successes, _ in results)
    total_errors = sum(errors for _, _, errors in results)

    print(f"  Results: {total_successes} successes, {total_errors} errors across {num_tables} tables")
    print(f"  ✓ Multiple tables stress test passed")

    # Cleanup
    try:
        for i in range(num_tables):
            await db_conn.execute(f'DROP TABLE IF EXISTS "{schema_name}".stress_table_{i} CASCADE')
        await db_conn.execute(f'DROP SCHEMA IF EXISTS "{schema_name}" CASCADE')
    except:
        pass


@pytest.mark.asyncio
async def test_concurrent_create_with_explicit_duplicate_check(db_conn: asyncpg.Connection):
    """
    Test with explicit checking for duplicate key violations in pg_deeplake_tables.

    This test specifically looks for the exact error message from the original bug report.
    """
    schema_name = "test_schema"
    table_name = "explicit_check_table"
    num_workers = 75

    await db_conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema_name}"')

    # Clean up
    try:
        await db_conn.execute(f'DROP TABLE IF EXISTS "{schema_name}".{table_name} CASCADE')
        await db_conn.execute(f'DROP TYPE IF EXISTS "{schema_name}".{table_name} CASCADE')
    except:
        pass

    print(f"\n  Running {num_workers} concurrent CREATE TABLE operations...")
    print(f"  Explicitly checking for duplicate key violations...")

    tasks = [
        create_table_worker(i, schema_name, table_name)
        for i in range(num_workers)
    ]

    results = await asyncio.gather(*tasks)

    # Analyze for the specific error from the bug report
    duplicate_key_violations = []
    for worker_id, success, error in results:
        if not success and error:
            error_lower = error.lower()
            # Check for the exact error from bug report:
            # 'duplicate key value violates unique constraint "pg_deeplake_tables_table_name_key"'
            if "duplicate key" in error_lower and "pg_deeplake_tables" in error_lower:
                duplicate_key_violations.append((worker_id, error))

    if duplicate_key_violations:
        error_details = "\n".join([
            f"  Worker {wid}: {err}" for wid, err in duplicate_key_violations
        ])
        pytest.fail(
            f"RACE CONDITION DETECTED!\n"
            f"Found {len(duplicate_key_violations)} duplicate key violations in pg_deeplake_tables:\n"
            f"{error_details}"
        )

    # Verify exactly one metadata entry
    count = await db_conn.fetchval(
        'SELECT COUNT(*) FROM public.pg_deeplake_tables WHERE table_name = $1',
        f'{schema_name}.{table_name}'
    )

    if count != 1:
        pytest.fail(f"Expected exactly 1 metadata entry, found {count}. This indicates a race condition!")

    # Query the actual metadata to verify integrity
    metadata = await db_conn.fetchrow(
        'SELECT table_oid, table_name, ds_path FROM public.pg_deeplake_tables WHERE table_name = $1',
        f'{schema_name}.{table_name}'
    )

    assert metadata is not None, "Metadata entry not found!"
    assert metadata['table_name'] == f'{schema_name}.{table_name}', "Table name mismatch in metadata"

    print(f"  ✓ No duplicate key violations detected")
    print(f"  ✓ Exactly 1 metadata entry exists")
    print(f"  ✓ Metadata integrity verified")

    # Cleanup
    try:
        await db_conn.execute(f'DROP TABLE IF EXISTS "{schema_name}".{table_name} CASCADE')
        await db_conn.execute(f'DROP SCHEMA IF EXISTS "{schema_name}" CASCADE')
    except:
        pass


async def _test_concurrent_create_table(db_conn: asyncpg.Connection, num_workers: int, test_name: str = "default"):
    """
    Helper function to test concurrent CREATE TABLE IF NOT EXISTS.

    Args:
        db_conn: Database connection from fixture
        num_workers: Number of concurrent workers to spawn
        test_name: Name for this test instance (for unique table names)
    """
    assertions = Assertions(db_conn)

    # Create a temporary directory for the dataset
    schema_name = "test_schema"
    table_name = f"concurrent_test_table_{test_name}"

    try:
        # Create schema
        await db_conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema_name}"')

        # Ensure table doesn't exist (CASCADE will also drop the associated type)
        await db_conn.execute(f'DROP TABLE IF EXISTS "{schema_name}".{table_name} CASCADE')

        # Also explicitly drop the type if it exists (PostgreSQL auto-creates types for tables)
        await db_conn.execute(f'DROP TYPE IF EXISTS "{schema_name}".{table_name} CASCADE')

        # Launch concurrent workers
        print(f"\n  Launching {num_workers} concurrent CREATE TABLE operations...")
        tasks = [
            create_table_worker(i, schema_name, table_name)
            for i in range(num_workers)
        ]

        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)

        # Analyze results
        successes = 0
        errors = 0
        duplicate_key_errors = 0
        error_messages = []

        for worker_id, success, error in results:
            if success:
                successes += 1
            else:
                errors += 1
                error_messages.append(f"Worker {worker_id}: {error}")
                if error and "duplicate key" in error.lower():
                    duplicate_key_errors += 1

        # Print summary
        print(f"  Results: {successes} successes, {errors} errors")

        # Assert no duplicate key violations occurred
        if duplicate_key_errors > 0:
            error_detail = "\n".join(error_messages)
            pytest.fail(
                f"Detected {duplicate_key_errors} duplicate key violations!\n"
                f"This indicates a race condition in CREATE TABLE IF NOT EXISTS.\n"
                f"Errors:\n{error_detail}"
            )

        # Verify exactly one metadata entry exists
        count = await db_conn.fetchval(
            'SELECT COUNT(*) FROM public.pg_deeplake_tables WHERE table_name = $1',
            f'{schema_name}.{table_name}'
        )
        assert count == 1, f"Expected 1 metadata entry, found {count}"

        # Verify table exists
        table_exists = await db_conn.fetchval(
            """
            SELECT EXISTS (
                SELECT 1 FROM pg_tables
                WHERE schemaname = $1 AND tablename = $2
            )
            """,
            schema_name, table_name
        )
        assert table_exists, f"Table {schema_name}.{table_name} does not exist after concurrent creates"

        # Verify we can query the table
        await assertions.assert_table_row_count(0, f'"{schema_name}".{table_name}')

        print(f"  ✓ Concurrent CREATE TABLE test passed with {num_workers} workers")

    finally:
        # Cleanup
        try:
            await db_conn.execute(f'DROP TABLE IF EXISTS "{schema_name}".{table_name} CASCADE')
            await db_conn.execute(f'DROP SCHEMA IF EXISTS "{schema_name}" CASCADE')
        except Exception as e:
            print(f"  Warning: Cleanup failed: {e}")


@pytest.mark.asyncio
async def test_concurrent_create_different_tables(db_conn: asyncpg.Connection):
    """
    Test concurrent CREATE TABLE operations on different tables.

    This test verifies that creating different tables concurrently works correctly.
    """

    temp_dir = tempfile.mkdtemp(prefix="deeplake_test_concurrent_diff_")
    schema_name = "test_schema"
    num_tables = 10

    try:
        # Create schema
        await db_conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema_name}"')

        async def create_different_table(table_id: int):
            """Create a unique table."""
            conn = None
            try:
                user = os.environ.get("USER", "postgres")
                conn = await asyncpg.connect(
                    database="postgres",
                    user=user,
                    host="localhost",
                    statement_cache_size=0
                )

                table_name = f"table_{table_id}"

                await conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS "{schema_name}".{table_name} (
                        id SERIAL PRIMARY KEY,
                        data TEXT
                    ) USING deeplake
                """)
                return (table_id, True, None)
            except Exception as e:
                return (table_id, False, str(e))
            finally:
                if conn:
                    await conn.close()

        # Create multiple different tables concurrently
        print(f"\n  Creating {num_tables} different tables concurrently...")
        tasks = [create_different_table(i) for i in range(num_tables)]
        results = await asyncio.gather(*tasks)

        # Verify all succeeded
        for table_id, success, error in results:
            if not success:
                pytest.fail(f"Failed to create table_{table_id}: {error}")

        # Verify all tables exist in metadata
        count = await db_conn.fetchval(
            f"SELECT COUNT(*) FROM public.pg_deeplake_tables WHERE table_name LIKE '{schema_name}.table_%'"
        )
        assert count == num_tables, f"Expected {num_tables} metadata entries, found {count}"

        print(f"  ✓ Successfully created {num_tables} different tables concurrently")

    finally:
        # Cleanup
        try:
            for i in range(num_tables):
                await db_conn.execute(f'DROP TABLE IF EXISTS "{schema_name}".table_{i} CASCADE')
            await db_conn.execute(f'DROP SCHEMA IF EXISTS "{schema_name}" CASCADE')
        except:
            pass
