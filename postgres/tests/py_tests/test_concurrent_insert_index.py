"""
Test concurrent insert operations with index creation.

This test verifies that:
1. Multiple clients can perform inserts concurrently while indexes are being created
2. Index creation can happen while inserts are in progress
3. All operations complete successfully without deadlocks or conflicts
"""
import pytest
import asyncpg
import asyncio
import os
import multiprocessing
import time
import subprocess
import signal
from lib.assertions import Assertions


# Number of insert iterations
INSERT_COUNT = 10000


def kill_postgres_server():
    """
    Forcibly kill all PostgreSQL server processes with SIGKILL.
    Called when server appears to have crashed to prevent hanging on shutdown.
    """
    try:
        # Find all postgres processes for current user
        user = os.environ.get("USER", "postgres")
        result = subprocess.run(
            ["pgrep", "-u", user, "postgres"],
            capture_output=True,
            text=True
        )

        if result.returncode == 0 and result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            print(f"  Found {len(pids)} PostgreSQL processes, killing with SIGKILL...")

            for pid in pids:
                try:
                    pid_int = int(pid)
                    os.kill(pid_int, signal.SIGKILL)
                    print(f"    Killed postgres process {pid}")
                except (ValueError, ProcessLookupError, PermissionError) as e:
                    print(f"    Could not kill process {pid}: {e}")
        else:
            print("  No PostgreSQL processes found to kill")

    except Exception as e:
        print(f"  Error while trying to kill PostgreSQL: {e}")


def insert_worker_process(insert_count: int, batch_size: int = 100) -> None:
    """
    Worker process that performs insert operations.
    Runs in separate process to avoid blocking.
    """
    async def do_inserts():
        user = os.environ.get("USER", "postgres")
        conn = await asyncpg.connect(
            database="postgres",
            user=user,
            host="localhost",
            statement_cache_size=0
        )

        try:
            print(f"Insert worker: Starting {insert_count} inserts in batches of {batch_size}...")

            completed = 0
            batch_num = 0

            while completed < insert_count:
                current_batch_size = min(batch_size, insert_count - completed)
                batch_num += 1

                # Single insert for first iteration
                if completed == 0:
                    await conn.execute("""
                        INSERT INTO chunk_text_image (id, text, image, metadata, chunk_index, file_id, created_at)
                        VALUES (
                            'chunk_12345',
                            'This is the text content of the chunk',
                            '\\xDEADBEEF'::bytea,
                            '{"key": "value", "tags": ["tag1", "tag2"]}'::jsonb,
                            1,
                            '550e8400-e29b-41d4-a716-446655440000'::uuid,
                            1700000000
                        )
                    """)
                    completed += 1
                    print(f"Insert worker: Completed 1 insert")
                    continue

                # Batch inserts
                values_list = []
                for i in range(current_batch_size):
                    idx = completed + i
                    chunk_id = f'chunk_{idx:09d}'
                    text = f'Chunk text content number {idx}'

                    if idx % 5 == 0:
                        image = 'NULL'
                    elif idx % 3 == 0:
                        image = "'\\x89504E47'::bytea"
                    else:
                        image = "'\\xDEADBEEF'::bytea"

                    metadata = f'{{"index": {idx}, "batch": {batch_num}}}'
                    chunk_index = idx
                    file_id = '550e8400-e29b-41d4-a716-446655440000' if idx % 2 == 0 else '660e8400-e29b-41d4-a716-446655440001'
                    created_at = 1700000000 + idx

                    values_list.append(
                        f"('{chunk_id}', '{text}', {image}, '{metadata}'::jsonb, "
                        f"{chunk_index}, '{file_id}'::uuid, {created_at})"
                    )

                values_str = ',\n                '.join(values_list)
                insert_sql = f"""
                    INSERT INTO chunk_text_image (id, text, image, metadata, chunk_index, file_id, created_at)
                    VALUES {values_str}
                """

                await conn.execute(insert_sql)
                completed += current_batch_size

                if completed % 1000 == 0:
                    print(f"Insert worker: Completed {completed}/{insert_count} inserts")

            # Final batch inserts
            await conn.execute("""
                INSERT INTO chunk_text_image (id, text, image, metadata, chunk_index, file_id, created_at)
                VALUES
                    ('chunk_final_001', 'First chunk', '\\x89504E47'::bytea, '{"page": 1}'::jsonb, 0, '550e8400-e29b-41d4-a716-446655440000'::uuid, 1700000000),
                    ('chunk_final_002', 'Second chunk', '\\x89504E48'::bytea, '{"page": 2}'::jsonb, 1, '550e8400-e29b-41d4-a716-446655440000'::uuid, 1700000001),
                    ('chunk_final_003', 'Third chunk', NULL, '{"page": 3}'::jsonb, 2, '550e8400-e29b-41d4-a716-446655440000'::uuid, 1700000002)
            """)

            print(f"Insert worker: Completed all {insert_count + 3} inserts successfully")

        finally:
            await conn.close()

    # Run async code in process
    asyncio.run(do_inserts())


def index_worker_process(delay_seconds: float = 0.0) -> None:
    """
    Worker process that creates indexes.
    Runs in separate process to avoid blocking.
    """
    async def do_indexes():
        time.sleep(delay_seconds)

        user = os.environ.get("USER", "postgres")
        conn = await asyncpg.connect(
            database="postgres",
            user=user,
            host="localhost",
            statement_cache_size=0
        )

        try:
            print("Index worker: Starting index creation...")

            print("Index worker: Creating exact_text index on file_id...")
            await conn.execute("""
                CREATE INDEX idx_default_chunk_text_image_file_id_exact_text
                ON chunk_text_image USING deeplake_index (file_id)
                WITH (index_type=exact_text)
            """)
            print("Index worker: ✓ Created exact_text index on file_id")

            print("Index worker: Creating exact_text index on id...")
            await conn.execute("""
                CREATE INDEX idx_default_chunk_text_image_id_exact_text
                ON chunk_text_image USING deeplake_index (id)
                WITH (index_type=exact_text)
            """)
            print("Index worker: ✓ Created exact_text index on id")

            print("Index worker: Creating bm25 index on text...")
            await conn.execute("""
                CREATE INDEX idx_default_chunk_text_image_text_bm25
                ON chunk_text_image USING deeplake_index (text)
                WITH (index_type=bm25)
            """)
            print("Index worker: ✓ Created bm25 index on text")

            print("Index worker: Completed all index creation successfully")

        finally:
            await conn.close()

    # Run async code in process
    asyncio.run(do_indexes())


@pytest.mark.asyncio
async def test_concurrent_insert_and_index_creation(db_conn: asyncpg.Connection):
    """
    Test concurrent insert operations with index creation using separate processes.

    This test:
    - Creates a table with domains for UUID and TEXT
    - Runs inserts in two separate processes
    - Creates indexes in another separate process (in parallel)
    - Verifies operations complete or timeout (indicating crash)
    """
    assertions = Assertions(db_conn)

    try:
        # Initialization: Create domains and table
        print("Setting up domains and table...")
        await db_conn.execute("CREATE DOMAIN file_id AS UUID")
        await db_conn.execute("CREATE DOMAIN chunk_id AS TEXT")

        await db_conn.execute("""
            CREATE TABLE chunk_text_image(
                id chunk_id,
                text text,
                image bytea,
                metadata jsonb,
                chunk_index bigint,
                file_id file_id,
                created_at bigint
            ) USING deeplake
        """)
        print("✓ Setup complete: domains and table created")

        # Start workers in separate processes
        print("\nStarting concurrent operations in separate processes...")

        insert_proc1 = multiprocessing.Process(
            target=insert_worker_process,
            args=(INSERT_COUNT,)
        )
        insert_proc2 = multiprocessing.Process(
            target=insert_worker_process,
            args=(INSERT_COUNT,)
        )
        index_proc = multiprocessing.Process(
            target=index_worker_process,
            args=(0.0,)
        )

        # Start all processes
        insert_proc1.start()
        insert_proc2.start()
        index_proc.start()

        # Monitor processes with timeout
        timeout = 30.0  # 30 second timeout
        start_time = time.time()

        processes = [
            ("Insert worker 1", insert_proc1),
            ("Insert worker 2", insert_proc2),
            ("Index worker", index_proc)
        ]

        # Wait for all processes to complete or timeout
        all_done = False
        while not all_done and (time.time() - start_time) < timeout:
            all_done = True
            for name, proc in processes:
                if proc.is_alive():
                    all_done = False

            if not all_done:
                time.sleep(0.5)

        # Check if we timed out
        if not all_done:
            print("\n⚠ Operations timed out - server likely crashed!")

            # Kill any remaining worker processes immediately with SIGKILL
            for name, proc in processes:
                if proc.is_alive():
                    print(f"  Killing {name} (PID: {proc.pid}) with SIGKILL")
                    proc.kill()  # Sends SIGKILL immediately
                    proc.join(timeout=1)  # Wait up to 1 second for process to die

            # Kill PostgreSQL server to prevent hanging on fixture cleanup
            print("\n  Killing PostgreSQL server to prevent hang on shutdown...")
            kill_postgres_server()
            time.sleep(1)  # Give processes time to die

            raise AssertionError(
                f"Test timed out after {timeout} seconds - PostgreSQL server likely crashed "
                "due to concurrent inserts and index creation"
            )

        # Check exit codes
        failures = []
        for name, proc in processes:
            proc.join()
            if proc.exitcode != 0:
                failures.append(f"{name} failed with exit code {proc.exitcode}")

        if failures:
            raise AssertionError("Worker processes failed:\n" + "\n".join(failures))

        print("\n✓ All workers completed successfully")

        # Verify server is still responsive
        try:
            await db_conn.fetchval("SELECT 1")
            print("✓ Server health check passed")
        except Exception as e:
            raise AssertionError(f"PostgreSQL server is not responsive: {e}")

        # Verify the results
        print("\nVerifying results...")

        # Check row count (2 * (INSERT_COUNT + 3 final inserts))
        expected_count = 2 * (INSERT_COUNT + 3)
        await assertions.assert_table_row_count(expected_count, "chunk_text_image")

        # Verify indexes exist
        index_count = await db_conn.fetchval("""
            SELECT COUNT(*) FROM pg_indexes
            WHERE tablename = 'chunk_text_image'
        """)
        assert index_count == 3, f"Expected 3 indexes, but found {index_count}"
        print("✓ All 3 indexes created successfully")

        # Verify indexes are valid
        invalid_indexes = await db_conn.fetch("""
            SELECT indexname FROM pg_indexes
            WHERE tablename = 'chunk_text_image'
            AND indexdef IS NULL
        """)
        assert len(invalid_indexes) == 0, f"Found invalid indexes: {invalid_indexes}"
        print("✓ All indexes are valid")

        # Test that indexes can be used for queries
        result = await db_conn.fetchval("""
            SELECT COUNT(*) FROM chunk_text_image
            WHERE file_id = '550e8400-e29b-41d4-a716-446655440000'::uuid
        """)
        assert result > 0, "No results found for file_id query"
        print(f"✓ file_id index query returned {result} rows")

        result = await db_conn.fetchval("""
            SELECT COUNT(*) FROM chunk_text_image
            WHERE id = 'chunk_12345'
        """)
        assert result == 2, "Expected 2 results for id query (from both workers)"
        print(f"✓ id index query returned {result} rows")

        result = await db_conn.fetchval("""
            SELECT COUNT(*) FROM chunk_text_image
            WHERE text ILIKE '%chunk%'
        """)
        assert result > 0, "No results found for text query"
        print(f"✓ text index query returned {result} rows")

        print("\n✓ Test passed: Concurrent inserts and index creation completed successfully")

    finally:
        # Cleanup - with aggressive timeout to avoid hanging if server crashed
        print("\nCleaning up...")

        async def do_cleanup():
            """Perform cleanup operations."""
            # First check if server is responsive
            try:
                await db_conn.fetchval("SELECT 1")
            except Exception as e:
                print(f"⚠ Server is not responsive: {e}")
                print("  Skipping cleanup - artifacts will be cleaned up on next test run")
                return

            # Server is alive, proceed with cleanup
            try:
                await db_conn.execute("DROP TABLE IF EXISTS chunk_text_image CASCADE")
                await db_conn.execute("DROP DOMAIN IF EXISTS chunk_id CASCADE")
                await db_conn.execute("DROP DOMAIN IF EXISTS file_id CASCADE")
                print("✓ Cleanup complete")
            except Exception as e:
                print(f"⚠ Cleanup failed: {e}")

        # Wrap entire cleanup in aggressive timeout
        cleanup_succeeded = False
        try:
            await asyncio.wait_for(do_cleanup(), timeout=5.0)
            cleanup_succeeded = True
        except asyncio.TimeoutError:
            print("⚠ Cleanup timed out after 5 seconds - server likely crashed")
        except Exception as e:
            print(f"⚠ Cleanup error: {e}")

        # If cleanup failed/timed out, try to forcibly close connection to prevent fixture cleanup hang
        if not cleanup_succeeded:
            try:
                # Try to terminate the connection without waiting
                db_conn.terminate()
                print("  Terminated database connection")
            except Exception:
                pass  # Ignore errors when terminating
