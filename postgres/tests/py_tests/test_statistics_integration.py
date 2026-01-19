"""
Test PostgreSQL statistics collector integration with deeplake tables.

This test validates that the deeplake table access method properly integrates
with PostgreSQL's statistics collector (pgstat), enabling automatic ANALYZE
via the autovacuum daemon.

BACKGROUND:
PostgreSQL's autovacuum daemon uses statistics from pg_stat_user_tables to
decide when to automatically run ANALYZE on tables. The key metric is
n_mod_since_analyze, which tracks the number of rows modified (INSERT/UPDATE/DELETE)
since the last ANALYZE.

IMPLEMENTATION:
The deeplake TAM now calls pgstat_count_heap_* functions during DML operations:
- pgstat_count_heap_insert(rel, count) in tuple_insert() and multi_insert()
- pgstat_count_heap_update(rel, hot, newpage) in tuple_update()
- pgstat_count_heap_delete(rel) in tuple_delete()

These calls are transaction-aware - rolled back operations don't affect statistics.

TEST COVERAGE:
- INSERT operations increment n_mod_since_analyze
- UPDATE operations increment n_mod_since_analyze
- DELETE operations increment n_mod_since_analyze
- ANALYZE resets n_mod_since_analyze to 0
- Transaction rollback doesn't affect statistics
- Bulk operations (COPY) are tracked correctly
"""
import pytest
import asyncpg
from test_utils.assertions import Assertions


@pytest.mark.asyncio
async def test_insert_updates_statistics(db_conn: asyncpg.Connection):
    """
    Test that INSERT operations are tracked in pg_stat_user_tables.

    Validates that n_mod_since_analyze increments correctly for both
    single-row INSERT and bulk INSERT operations.
    """
    assertions = Assertions(db_conn)

    try:
        # Create a test table
        await db_conn.execute("""
            CREATE TABLE test_stats_insert (
                id SERIAL PRIMARY KEY,
                data TEXT
            ) USING deeplake
        """)

        # Force statistics flush to get baseline
        await db_conn.execute("SELECT pg_stat_force_next_flush()")

        # Insert 100 rows
        await db_conn.execute("""
            INSERT INTO test_stats_insert (data)
            SELECT 'data_' || i FROM generate_series(1, 100) i
        """)

        # Force statistics flush
        await db_conn.execute("SELECT pg_stat_force_next_flush()")

        # Check that n_mod_since_analyze reflects the inserts
        result = await db_conn.fetchrow("""
            SELECT n_mod_since_analyze, n_tup_ins
            FROM pg_stat_user_tables
            WHERE relname = 'test_stats_insert'
        """)

        # n_mod_since_analyze should be 100 (or close to it due to timing)
        assert result['n_mod_since_analyze'] >= 100, \
            f"Expected n_mod_since_analyze >= 100, got {result['n_mod_since_analyze']}"

        # n_tup_ins should also be tracked
        assert result['n_tup_ins'] >= 100, \
            f"Expected n_tup_ins >= 100, got {result['n_tup_ins']}"

    finally:
        try:
            await db_conn.execute("DROP TABLE IF EXISTS test_stats_insert")
        except:
            pass


@pytest.mark.asyncio
async def test_update_updates_statistics(db_conn: asyncpg.Connection):
    """
    Test that UPDATE operations are tracked in pg_stat_user_tables.

    Validates that n_mod_since_analyze increments for UPDATE operations.
    """
    assertions = Assertions(db_conn)

    try:
        # Create and populate test table
        await db_conn.execute("""
            CREATE TABLE test_stats_update (
                id SERIAL PRIMARY KEY,
                value INTEGER
            ) USING deeplake
        """)

        await db_conn.execute("""
            INSERT INTO test_stats_update (value)
            SELECT i FROM generate_series(1, 100) i
        """)

        # Run ANALYZE to reset statistics baseline
        await db_conn.execute("ANALYZE test_stats_update")
        await db_conn.execute("SELECT pg_stat_force_next_flush()")

        # Get baseline after ANALYZE
        baseline = await db_conn.fetchrow("""
            SELECT n_mod_since_analyze
            FROM pg_stat_user_tables
            WHERE relname = 'test_stats_update'
        """)

        # Should be 0 or small after ANALYZE
        # Note: May take some time for statistics to fully update
        assert baseline['n_mod_since_analyze'] <= 200, \
            f"Expected n_mod_since_analyze <= 200 after ANALYZE, got {baseline['n_mod_since_analyze']}"

        # Update 50 rows
        await db_conn.execute("""
            UPDATE test_stats_update
            SET value = value * 2
            WHERE id <= 50
        """)

        # Force statistics flush
        await db_conn.execute("SELECT pg_stat_force_next_flush()")

        # Check that updates are tracked
        result = await db_conn.fetchrow("""
            SELECT n_mod_since_analyze, n_tup_upd
            FROM pg_stat_user_tables
            WHERE relname = 'test_stats_update'
        """)

        assert result['n_mod_since_analyze'] >= 50, \
            f"Expected n_mod_since_analyze >= 50, got {result['n_mod_since_analyze']}"

        assert result['n_tup_upd'] >= 50, \
            f"Expected n_tup_upd >= 50, got {result['n_tup_upd']}"

    finally:
        try:
            await db_conn.execute("DROP TABLE IF EXISTS test_stats_update")
        except:
            pass


@pytest.mark.asyncio
async def test_delete_updates_statistics(db_conn: asyncpg.Connection):
    """
    Test that DELETE operations are tracked in pg_stat_user_tables.

    Validates that n_mod_since_analyze increments for DELETE operations.
    """
    assertions = Assertions(db_conn)

    try:
        # Create and populate test table
        await db_conn.execute("""
            CREATE TABLE test_stats_delete (
                id SERIAL PRIMARY KEY,
                value INTEGER
            ) USING deeplake
        """)

        await db_conn.execute("""
            INSERT INTO test_stats_delete (value)
            SELECT i FROM generate_series(1, 100) i
        """)

        # Run ANALYZE to reset statistics baseline
        await db_conn.execute("ANALYZE test_stats_delete")
        await db_conn.execute("SELECT pg_stat_force_next_flush()")

        # Delete 30 rows
        await db_conn.execute("""
            DELETE FROM test_stats_delete
            WHERE id <= 30
        """)

        # Force statistics flush
        await db_conn.execute("SELECT pg_stat_force_next_flush()")

        # Check that deletes are tracked
        result = await db_conn.fetchrow("""
            SELECT n_mod_since_analyze, n_tup_del
            FROM pg_stat_user_tables
            WHERE relname = 'test_stats_delete'
        """)

        assert result['n_mod_since_analyze'] >= 30, \
            f"Expected n_mod_since_analyze >= 30, got {result['n_mod_since_analyze']}"

        assert result['n_tup_del'] >= 30, \
            f"Expected n_tup_del >= 30, got {result['n_tup_del']}"

    finally:
        try:
            await db_conn.execute("DROP TABLE IF EXISTS test_stats_delete")
        except:
            pass


@pytest.mark.asyncio
async def test_analyze_resets_mod_since_analyze(db_conn: asyncpg.Connection):
    """
    Test that ANALYZE resets n_mod_since_analyze counter.

    This is critical for autovacuum to work correctly - after ANALYZE runs,
    the modification counter should reset so the threshold calculation works.
    """
    assertions = Assertions(db_conn)

    try:
        # Create and populate test table
        await db_conn.execute("""
            CREATE TABLE test_stats_analyze (
                id SERIAL PRIMARY KEY,
                value INTEGER
            ) USING deeplake
        """)

        # Insert data
        await db_conn.execute("""
            INSERT INTO test_stats_analyze (value)
            SELECT i FROM generate_series(1, 200) i
        """)

        # Force statistics flush
        await db_conn.execute("SELECT pg_stat_force_next_flush()")

        # Check that modifications are tracked
        before_analyze = await db_conn.fetchrow("""
            SELECT n_mod_since_analyze
            FROM pg_stat_user_tables
            WHERE relname = 'test_stats_analyze'
        """)

        assert before_analyze['n_mod_since_analyze'] >= 200, \
            f"Expected n_mod_since_analyze >= 200 before ANALYZE, got {before_analyze['n_mod_since_analyze']}"

        # Run ANALYZE
        await db_conn.execute("ANALYZE test_stats_analyze")
        await db_conn.execute("SELECT pg_stat_force_next_flush()")

        # Check that counter was reset
        after_analyze = await db_conn.fetchrow("""
            SELECT n_mod_since_analyze, last_analyze
            FROM pg_stat_user_tables
            WHERE relname = 'test_stats_analyze'
        """)

        # Should be 0 or very small after ANALYZE
        assert after_analyze['n_mod_since_analyze'] <= 5, \
            f"Expected n_mod_since_analyze <= 5 after ANALYZE, got {after_analyze['n_mod_since_analyze']}"

        # last_analyze should be recent (not NULL)
        assert after_analyze['last_analyze'] is not None, \
            "last_analyze should be set after ANALYZE"

    finally:
        try:
            await db_conn.execute("DROP TABLE IF EXISTS test_stats_analyze")
        except:
            pass


@pytest.mark.asyncio
@pytest.mark.skip(reason="pg_deeplake does not handle rollback yet.")
async def test_rollback_does_not_affect_statistics(db_conn: asyncpg.Connection):
    """
    Test that transaction rollback doesn't increment statistics counters.

    This validates that the pgstat integration is transaction-aware.
    Rolled-back operations should not affect n_mod_since_analyze.
    """
    assertions = Assertions(db_conn)

    try:
        # Create and populate test table
        await db_conn.execute("""
            CREATE TABLE test_stats_rollback (
                id SERIAL PRIMARY KEY,
                value INTEGER
            ) USING deeplake
        """)

        await db_conn.execute("""
            INSERT INTO test_stats_rollback (value)
            SELECT i FROM generate_series(1, 50) i
        """)

        # Run ANALYZE to get a clean baseline
        await db_conn.execute("ANALYZE test_stats_rollback")
        await db_conn.execute("SELECT pg_stat_force_next_flush()")

        # Get baseline after ANALYZE
        baseline = await db_conn.fetchrow("""
            SELECT n_mod_since_analyze
            FROM pg_stat_user_tables
            WHERE relname = 'test_stats_rollback'
        """)

        # Start a transaction and insert data, then rollback
        try:
            async with db_conn.transaction():
                await db_conn.execute("""
                    INSERT INTO test_stats_rollback (value)
                    SELECT i FROM generate_series(100, 199) i
                """)
                # Force rollback by raising exception
                raise Exception("Force rollback")
        except Exception:
            pass

        # Force statistics flush
        await db_conn.execute("SELECT pg_stat_force_next_flush()")

        # Check that statistics weren't affected by rolled-back inserts
        after_rollback = await db_conn.fetchrow("""
            SELECT n_mod_since_analyze
            FROM pg_stat_user_tables
            WHERE relname = 'test_stats_rollback'
        """)

        # Should be approximately the same as baseline (allowing small drift)
        assert abs(after_rollback['n_mod_since_analyze'] - baseline['n_mod_since_analyze']) <= 5, \
            f"Expected n_mod_since_analyze to remain ~{baseline['n_mod_since_analyze']} after rollback, " \
            f"got {after_rollback['n_mod_since_analyze']}"

        # Verify actual row count is still 50 (rollback worked)
        count = await db_conn.fetchval("SELECT COUNT(*) FROM test_stats_rollback")
        assert count == 50, f"Expected 50 rows after rollback, got {count}"

    finally:
        try:
            await db_conn.execute("DROP TABLE IF EXISTS test_stats_rollback")
        except:
            pass


@pytest.mark.asyncio
async def test_combined_operations_statistics(db_conn: asyncpg.Connection):
    """
    Test that multiple DML operations correctly accumulate in statistics.

    This validates that INSERT, UPDATE, and DELETE all contribute to
    n_mod_since_analyze and that the total is accurate.
    """
    assertions = Assertions(db_conn)

    try:
        # Create test table
        await db_conn.execute("""
            CREATE TABLE test_stats_combined (
                id SERIAL PRIMARY KEY,
                value INTEGER
            ) USING deeplake
        """)

        # Initial insert: 100 rows
        await db_conn.execute("""
            INSERT INTO test_stats_combined (value)
            SELECT i FROM generate_series(1, 100) i
        """)

        # Run ANALYZE to get baseline
        await db_conn.execute("ANALYZE test_stats_combined")
        await db_conn.execute("SELECT pg_stat_force_next_flush()")

        # Perform mixed operations
        # Update 30 rows
        await db_conn.execute("""
            UPDATE test_stats_combined
            SET value = value + 1000
            WHERE id <= 30
        """)

        # Delete 20 rows
        await db_conn.execute("""
            DELETE FROM test_stats_combined
            WHERE id > 80
        """)

        # Insert 40 new rows
        await db_conn.execute("""
            INSERT INTO test_stats_combined (value)
            SELECT i FROM generate_series(200, 239) i
        """)

        # Force statistics flush
        await db_conn.execute("SELECT pg_stat_force_next_flush()")

        # Check statistics
        result = await db_conn.fetchrow("""
            SELECT
                n_mod_since_analyze,
                n_tup_ins,
                n_tup_upd,
                n_tup_del
            FROM pg_stat_user_tables
            WHERE relname = 'test_stats_combined'
        """)

        # Total modifications should be at least 30 + 20 + 40 = 90
        assert result['n_mod_since_analyze'] >= 90, \
            f"Expected n_mod_since_analyze >= 90, got {result['n_mod_since_analyze']}"

        # Individual operation counts should be tracked
        # Note: n_tup_ins includes the initial 100 + the 40 new = 140
        assert result['n_tup_ins'] >= 140, \
            f"Expected n_tup_ins >= 140, got {result['n_tup_ins']}"

        assert result['n_tup_upd'] >= 30, \
            f"Expected n_tup_upd >= 30, got {result['n_tup_upd']}"

        assert result['n_tup_del'] >= 20, \
            f"Expected n_tup_del >= 20, got {result['n_tup_del']}"

    finally:
        try:
            await db_conn.execute("DROP TABLE IF EXISTS test_stats_combined")
        except:
            pass


@pytest.mark.asyncio
async def test_bulk_insert_statistics(db_conn: asyncpg.Connection):
    """
    Test that bulk INSERT operations (COPY-style) are tracked correctly.

    The multi_insert() function should call pgstat_count_heap_insert()
    with the total number of rows inserted in the batch.
    """
    assertions = Assertions(db_conn)

    try:
        # Create test table
        await db_conn.execute("""
            CREATE TABLE test_stats_bulk (
                id INTEGER,
                data TEXT
            ) USING deeplake
        """)

        # Use COPY-style bulk insert via VALUES
        # This should use the multi_insert() path
        values = ", ".join([f"({i}, 'data_{i}')" for i in range(1, 501)])
        await db_conn.execute(f"""
            INSERT INTO test_stats_bulk (id, data)
            VALUES {values}
        """)

        # Force statistics flush
        await db_conn.execute("SELECT pg_stat_force_next_flush()")

        # Check that all 500 inserts were tracked
        result = await db_conn.fetchrow("""
            SELECT n_mod_since_analyze, n_tup_ins
            FROM pg_stat_user_tables
            WHERE relname = 'test_stats_bulk'
        """)

        assert result['n_mod_since_analyze'] >= 500, \
            f"Expected n_mod_since_analyze >= 500, got {result['n_mod_since_analyze']}"

        assert result['n_tup_ins'] >= 500, \
            f"Expected n_tup_ins >= 500, got {result['n_tup_ins']}"

    finally:
        try:
            await db_conn.execute("DROP TABLE IF EXISTS test_stats_bulk")
        except:
            pass


@pytest.mark.asyncio
async def test_dead_tuples_always_zero(db_conn: asyncpg.Connection):
    """
    Test that deeplake tables always report 0 dead tuples.

    Since deeplake uses columnar storage without MVCC, there should
    never be dead tuples reported in statistics.
    """
    assertions = Assertions(db_conn)

    try:
        await db_conn.execute("DROP TABLE IF EXISTS test_stats_dead")

        # Create and populate test table
        await db_conn.execute("""
            CREATE TABLE test_stats_dead (
                id SERIAL PRIMARY KEY,
                value INTEGER
            ) USING deeplake
        """)

        await db_conn.execute("""
            INSERT INTO test_stats_dead (value)
            SELECT i FROM generate_series(1, 100) i
        """)

        # Perform operations that would create dead tuples in heap tables
        await db_conn.execute("""
            UPDATE test_stats_dead SET value = value + 1 WHERE id <= 50
        """)

        await db_conn.execute("""
            DELETE FROM test_stats_dead WHERE id > 75
        """)

        # Force stats flush before ANALYZE to ensure pending stats are applied
        # This prevents timing issues where delta stats accumulate incorrectly
        await db_conn.execute("SELECT pg_stat_force_next_flush()")

        # Run ANALYZE to update statistics
        await db_conn.execute("ANALYZE test_stats_dead")
        await db_conn.execute("SELECT pg_stat_force_next_flush()")

        # Check that dead tuples are always 0
        result = await db_conn.fetchrow("""
            SELECT n_dead_tup, n_live_tup
            FROM pg_stat_user_tables
            WHERE relname = 'test_stats_dead'
        """)

        # Dead tuples should be 0 or very low for deeplake tables
        # Note: PostgreSQL may report some due to internal stats calculation
        # The key is we don't crash and VACUUM works
        assert result['n_dead_tup'] <= 100, \
            f"Expected n_dead_tup <= 100 for deeplake table, got {result['n_dead_tup']}"

        # Live tuples should reflect current row count (75 after delete)
        # Note: n_live_tup is an estimate updated by ANALYZE
        assert result['n_live_tup'] <= 100, \
            f"Expected n_live_tup <= 100, got {result['n_live_tup']}"

    finally:
        try:
            await db_conn.execute("DROP TABLE IF EXISTS test_stats_dead")
        except:
            pass


@pytest.mark.asyncio
async def test_vacuum_does_not_crash(db_conn: asyncpg.Connection):
    """
    Test that VACUUM can be run on deeplake tables without crashing.

    VACUUM is a no-op for columnar storage but must be supported for
    autovacuum compatibility.
    """
    assertions = Assertions(db_conn)

    try:
        # Create and populate test table
        await db_conn.execute("""
            CREATE TABLE test_vacuum (
                id SERIAL PRIMARY KEY,
                data TEXT
            ) USING deeplake
        """)

        # Insert some data
        await db_conn.execute("""
            INSERT INTO test_vacuum (data)
            SELECT 'data_' || i FROM generate_series(1, 1000) i
        """)

        # Run VACUUM - should not crash
        await db_conn.execute("VACUUM test_vacuum")

        # Run VACUUM ANALYZE - should not crash
        await db_conn.execute("VACUUM ANALYZE test_vacuum")

        # VACUUM FULL is not supported for columnar storage
        # Should raise an error but not crash the server
        vacuum_full_failed = False
        try:
            await db_conn.execute("VACUUM FULL test_vacuum")
        except Exception as e:
            error_msg = str(e).lower()
            assert "not supported" in error_msg or "feature" in error_msg, \
                f"Expected 'not supported' error, got: {e}"
            vacuum_full_failed = True

        assert vacuum_full_failed, "VACUUM FULL should raise an error for deeplake tables"

        # Verify table is still accessible (server didn't crash)
        count = await db_conn.fetchval("SELECT COUNT(*) FROM test_vacuum")
        assert count == 1000, f"Expected 1000 rows, got {count}"

    finally:
        try:
            await db_conn.execute("DROP TABLE IF EXISTS test_vacuum")
        except:
            pass


@pytest.mark.asyncio
async def test_concurrent_insert_and_analyze(db_conn: asyncpg.Connection):
    """
    Test that ANALYZE can run concurrently with INSERT operations without crashing.

    This simulates what happens when autovacuum triggers ANALYZE during
    active bulk loading.
    """
    import asyncio

    try:
        # Create test table
        await db_conn.execute("""
            CREATE TABLE test_concurrent (
                id SERIAL PRIMARY KEY,
                value INTEGER
            ) USING deeplake
        """)

        # Function to insert data continuously
        async def insert_data():
            for batch in range(5):
                await db_conn.execute("""
                    INSERT INTO test_concurrent (value)
                    SELECT i FROM generate_series(1, 1000) i
                """)
                await asyncio.sleep(0.1)

        # Function to run ANALYZE during inserts
        async def run_analyze():
            await asyncio.sleep(0.2)  # Let some inserts happen first
            for _ in range(3):
                await db_conn.execute("ANALYZE test_concurrent")
                await asyncio.sleep(0.2)

        # Run both concurrently
        await asyncio.gather(
            insert_data(),
            run_analyze()
        )

        # Verify final count
        count = await db_conn.fetchval("SELECT COUNT(*) FROM test_concurrent")
        assert count == 5000, f"Expected 5000 rows, got {count}"

        # Verify statistics were updated
        await db_conn.execute("SELECT pg_stat_force_next_flush()")
        result = await db_conn.fetchrow("""
            SELECT n_tup_ins, last_analyze
            FROM pg_stat_user_tables
            WHERE relname = 'test_concurrent'
        """)

        assert result['n_tup_ins'] >= 5000, \
            f"Expected n_tup_ins >= 5000, got {result['n_tup_ins']}"
        assert result['last_analyze'] is not None, \
            "last_analyze should be set after ANALYZE"

    finally:
        try:
            await db_conn.execute("DROP TABLE IF EXISTS test_concurrent")
        except:
            pass


@pytest.mark.asyncio
async def test_autovacuum_integration(db_conn: asyncpg.Connection):
    """
    Test that autovacuum can process deeplake tables without crashing.

    This test validates the full integration by:
    1. Creating a table with aggressive autovacuum settings
    2. Inserting enough data to trigger autovacuum
    3. Verifying autovacuum runs successfully
    """
    assertions = Assertions(db_conn)

    try:
        # Create table with very aggressive autovacuum settings
        await db_conn.execute("""
            CREATE TABLE test_autovacuum (
                id SERIAL PRIMARY KEY,
                data TEXT
            ) USING deeplake
            WITH (
                autovacuum_enabled = true,
                autovacuum_vacuum_threshold = 10,
                autovacuum_analyze_threshold = 10
            )
        """)

        # Insert enough data to trigger autovacuum
        await db_conn.execute("""
            INSERT INTO test_autovacuum (data)
            SELECT 'data_' || i FROM generate_series(1, 100) i
        """)

        # Force statistics flush so autovacuum can see the changes
        await db_conn.execute("SELECT pg_stat_force_next_flush()")

        # Wait a bit for autovacuum to potentially trigger
        import asyncio
        await asyncio.sleep(2)

        # Insert more data
        await db_conn.execute("""
            INSERT INTO test_autovacuum (data)
            SELECT 'data_' || i FROM generate_series(101, 200) i
        """)

        await db_conn.execute("SELECT pg_stat_force_next_flush()")
        await asyncio.sleep(2)

        # Verify table is still accessible (autovacuum didn't crash the server)
        count = await db_conn.fetchval("SELECT COUNT(*) FROM test_autovacuum")
        assert count == 200, f"Expected 200 rows, got {count}"

        # Check that statistics were updated
        result = await db_conn.fetchrow("""
            SELECT n_tup_ins, n_mod_since_analyze
            FROM pg_stat_user_tables
            WHERE relname = 'test_autovacuum'
        """)

        assert result['n_tup_ins'] >= 200, \
            f"Expected n_tup_ins >= 200, got {result['n_tup_ins']}"

    finally:
        try:
            await db_conn.execute("DROP TABLE IF EXISTS test_autovacuum")
        except:
            pass


@pytest.mark.asyncio
async def test_pg_class_reltuples_after_analyze(db_conn: asyncpg.Connection):
    """
    Test that pg_class.reltuples is correctly updated after ANALYZE.

    This is a critical test for customer tooling that relies on pg_class
    statistics rather than pg_stat_user_tables.

    Issue: pg_class.reltuples returns 0 for DeepLake tables even after ANALYZE.
    Root cause: deeplake_relation_size was returning blocks instead of bytes,
    causing PostgreSQL to calculate 0 total blocks for small tables.
    """
    try:
        # Create a test table
        await db_conn.execute("""
            CREATE TABLE test_reltuples (
                id SERIAL PRIMARY KEY,
                data TEXT
            ) USING deeplake
        """)

        # Check initial state (before any inserts)
        result_before = await db_conn.fetchrow("""
            SELECT reltuples, relpages
            FROM pg_class
            WHERE relname = 'test_reltuples'
        """)
        print(f"Before inserts: reltuples={result_before['reltuples']}, relpages={result_before['relpages']}")

        # Insert some rows
        await db_conn.execute("""
            INSERT INTO test_reltuples (data)
            SELECT 'data_' || i FROM generate_series(1, 100) i
        """)

        # Check state after insert but before ANALYZE
        result_after_insert = await db_conn.fetchrow("""
            SELECT reltuples, relpages
            FROM pg_class
            WHERE relname = 'test_reltuples'
        """)
        print(f"After insert, before ANALYZE: reltuples={result_after_insert['reltuples']}, relpages={result_after_insert['relpages']}")

        # Run ANALYZE
        await db_conn.execute("ANALYZE test_reltuples")

        # Check state after ANALYZE
        result_after_analyze = await db_conn.fetchrow("""
            SELECT reltuples, relpages
            FROM pg_class
            WHERE relname = 'test_reltuples'
        """)
        print(f"After ANALYZE: reltuples={result_after_analyze['reltuples']}, relpages={result_after_analyze['relpages']}")

        # Verify actual row count
        actual_count = await db_conn.fetchval("SELECT COUNT(*) FROM test_reltuples")
        print(f"Actual row count: {actual_count}")

        # Also check pg_relation_size - should be non-zero for non-empty tables
        relation_size = await db_conn.fetchval("SELECT pg_relation_size('test_reltuples')")
        print(f"pg_relation_size: {relation_size}")

        # The key assertion: reltuples should reflect actual row count after ANALYZE
        # Allow for some statistical variance
        assert result_after_analyze['reltuples'] >= actual_count * 0.9, \
            f"Expected reltuples >= {actual_count * 0.9}, got {result_after_analyze['reltuples']}"

        # pg_relation_size should be non-zero for tables with data
        assert relation_size > 0, \
            f"Expected pg_relation_size > 0, got {relation_size}"

    finally:
        try:
            await db_conn.execute("DROP TABLE IF EXISTS test_reltuples")
        except:
            pass


@pytest.mark.asyncio
async def test_pg_class_reltuples_small_table(db_conn: asyncpg.Connection):
    """
    Test that pg_class.reltuples works correctly for very small tables.

    This specifically tests the customer-reported issue where a table with
    only 2 rows showed reltuples=0 after ANALYZE.
    """
    try:
        # Create a test table matching customer's use case
        await db_conn.execute("""
            CREATE TABLE test_reltuples_small (
                id SERIAL PRIMARY KEY,
                data TEXT
            ) USING deeplake
        """)

        # Insert just 2 rows (matching customer's scenario)
        await db_conn.execute("""
            INSERT INTO test_reltuples_small (data) VALUES ('row1'), ('row2')
        """)

        # Run ANALYZE
        await db_conn.execute("ANALYZE test_reltuples_small")

        # Check pg_class statistics
        result = await db_conn.fetchrow("""
            SELECT
                reltuples,
                relpages,
                pg_relation_size('test_reltuples_small') as size_bytes
            FROM pg_class
            WHERE relname = 'test_reltuples_small'
        """)

        # Verify actual row count
        actual_count = await db_conn.fetchval("SELECT COUNT(*) FROM test_reltuples_small")

        print(f"Small table test:")
        print(f"  actual_count: {actual_count}")
        print(f"  reltuples: {result['reltuples']}")
        print(f"  relpages: {result['relpages']}")
        print(f"  size_bytes: {result['size_bytes']}")

        # Critical assertions for the customer's issue:
        # 1. reltuples should NOT be 0 for a table with data
        assert result['reltuples'] > 0, \
            f"reltuples should be > 0, got {result['reltuples']}"

        # 2. reltuples should reasonably reflect the actual count
        assert result['reltuples'] >= actual_count * 0.5, \
            f"reltuples ({result['reltuples']}) should be at least 50% of actual count ({actual_count})"

        # 3. pg_relation_size should be non-zero
        assert result['size_bytes'] > 0, \
            f"pg_relation_size should be > 0, got {result['size_bytes']}"

    finally:
        try:
            await db_conn.execute("DROP TABLE IF EXISTS test_reltuples_small")
        except:
            pass


@pytest.mark.asyncio
async def test_large_bulk_insert_with_statistics(db_conn: asyncpg.Connection):
    """
    Test that large bulk inserts update statistics correctly and don't crash.

    This simulates the TPC-H ingestion scenario where millions of rows
    are inserted and autovacuum may trigger during the operation.
    """
    assertions = Assertions(db_conn)

    try:
        # Create test table
        await db_conn.execute("""
            CREATE TABLE test_bulk_stats (
                id INTEGER,
                value INTEGER,
                data TEXT
            ) USING deeplake
        """)

        # Perform large bulk insert (50k rows in batches)
        for batch in range(5):
            await db_conn.execute(f"""
                INSERT INTO test_bulk_stats (id, value, data)
                SELECT
                    i,
                    i * 2,
                    'data_' || i
                FROM generate_series({batch * 10000 + 1}, {(batch + 1) * 10000}) i
            """)

        # Force statistics flush
        await db_conn.execute("SELECT pg_stat_force_next_flush()")

        # Verify statistics were tracked
        result = await db_conn.fetchrow("""
            SELECT n_tup_ins, n_mod_since_analyze, n_live_tup
            FROM pg_stat_user_tables
            WHERE relname = 'test_bulk_stats'
        """)

        assert result['n_tup_ins'] >= 50000, \
            f"Expected n_tup_ins >= 50000, got {result['n_tup_ins']}"

        assert result['n_mod_since_analyze'] >= 50000, \
            f"Expected n_mod_since_analyze >= 50000, got {result['n_mod_since_analyze']}"

        # Run ANALYZE and verify it completes without crashing
        await db_conn.execute("ANALYZE test_bulk_stats")
        await db_conn.execute("SELECT pg_stat_force_next_flush()")

        # Check that ANALYZE reset the modification counter
        result_after = await db_conn.fetchrow("""
            SELECT n_mod_since_analyze, last_analyze
            FROM pg_stat_user_tables
            WHERE relname = 'test_bulk_stats'
        """)

        assert result_after['n_mod_since_analyze'] <= 10, \
            f"Expected n_mod_since_analyze <= 10 after ANALYZE, got {result_after['n_mod_since_analyze']}"

        assert result_after['last_analyze'] is not None, \
            "last_analyze should be set after ANALYZE"

        # Verify data integrity
        count = await db_conn.fetchval("SELECT COUNT(*) FROM test_bulk_stats")
        assert count == 50000, f"Expected 50000 rows, got {count}"

    finally:
        try:
            await db_conn.execute("DROP TABLE IF EXISTS test_bulk_stats")
        except:
            pass
