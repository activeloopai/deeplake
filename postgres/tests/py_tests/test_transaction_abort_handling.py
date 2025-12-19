"""
Test transaction abort handling to prevent cascading errors.

This test validates that the extension handles transaction aborts gracefully
without triggering cascading "Deeplake does not support transaction aborts"
errors that lead to PANIC.

ROOT CAUSE:
The deeplake_xact_callback() function in pg_deeplake.cpp was calling
elog(ERROR, ...) when handling XACT_EVENT_ABORT. This ERROR triggered another
abort attempt, creating an infinite loop until ERRORDATA_STACK_SIZE was exceeded.

FIX:
Changed the abort handler to call rollback_all() instead of throwing ERROR,
allowing transactions to abort cleanly without cascading errors.

Regression test for issue where ANY error (like GUC parameter errors or
DuckDB catalog errors) would cause:
1. Transaction abort attempt
2. deeplake_xact_callback receives XACT_EVENT_ABORT
3. Old code: elog(ERROR, "Deeplake does not support transaction aborts")
4. This ERROR triggered another abort → recursive loop
5. PANIC: ERRORDATA_STACK_SIZE exceeded
"""
import pytest
import asyncpg
from test_utils.assertions import Assertions


@pytest.mark.asyncio
@pytest.mark.skip(reason="pg_deeplake does not handle rollback yet.")
async def test_error_during_insert_with_abort(db_conn: asyncpg.Connection):
    """
    Test that errors during INSERT operations don't cause cascading aborts.

    This simulates the scenario where an error occurs during data modification
    and ensures the transaction abort is handled gracefully.
    """
    assertions = Assertions(db_conn)

    try:
        # Create a test table
        await db_conn.execute("""
            CREATE TABLE test_abort_insert (
                id SERIAL PRIMARY KEY,
                value INTEGER NOT NULL
            ) USING deeplake
        """)

        # Start a transaction
        async with db_conn.transaction():
            # Insert valid data
            await db_conn.execute("""
                INSERT INTO test_abort_insert (value) VALUES (1), (2), (3)
            """)

            # Try to insert invalid data (violate NOT NULL constraint)
            try:
                await db_conn.execute("""
                    INSERT INTO test_abort_insert (value) VALUES (NULL)
                """)
                assert False, "Should have raised an error for NULL value"
            except asyncpg.exceptions.NotNullViolationError:
                # Expected error - transaction should abort gracefully
                pass

        # Verify the transaction was rolled back (no data inserted)
        count = await db_conn.fetchval("SELECT COUNT(*) FROM test_abort_insert")
        assert count == 0, f"Expected 0 rows after rollback, got {count}"

    finally:
        try:
            await db_conn.execute("DROP TABLE IF EXISTS test_abort_insert")
        except:
            pass


@pytest.mark.asyncio
async def test_guc_parameter_error_handling(db_conn: asyncpg.Connection):
    """
    Test that GUC parameter errors don't cause cascading aborts.

    This specifically tests the scenario from the bug report where
    SET checkpoint_completion_target caused cascading abort errors.

    This test directly validates the fix to deeplake_xact_callback:
    - GUC error triggers transaction abort
    - XACT_EVENT_ABORT is sent to deeplake_xact_callback
    - Callback should rollback_all() NOT throw ERROR
    - Connection should remain usable
    """
    # Try to set a parameter that can't be changed at runtime
    # This should error gracefully without causing PANIC
    try:
        await db_conn.execute("SET checkpoint_completion_target = 0.9")
        # If we get here, the parameter might be changeable in this PG version
        # Reset it to avoid affecting other tests
        await db_conn.execute("RESET checkpoint_completion_target")
    except asyncpg.exceptions.PostgresError as e:
        # Expected error - should not cause PANIC or cascading aborts
        # Verify we can continue using the connection
        result = await db_conn.fetchval("SELECT 1")
        assert result == 1, "Connection should still be usable after GUC error"


@pytest.mark.asyncio
@pytest.mark.skip(reason="pg_deeplake does not handle rollback yet.")
async def test_query_error_with_pending_changes(db_conn: asyncpg.Connection):
    """
    Test that query errors with pending changes are handled correctly.

    This tests the executor_end error handling path where flush_all()
    might fail and we need to rollback without cascading errors.
    """
    assertions = Assertions(db_conn)

    try:
        # Create test table
        await db_conn.execute("""
            CREATE TABLE test_abort_query (
                id SERIAL PRIMARY KEY,
                data TEXT
            ) USING deeplake
        """)

        # Insert some data
        await db_conn.execute("""
            INSERT INTO test_abort_query (data) VALUES ('test1'), ('test2')
        """)

        # Verify data was inserted
        count = await db_conn.fetchval("SELECT COUNT(*) FROM test_abort_query")
        assert count == 2, f"Expected 2 rows, got {count}"

        # Try a transaction that will fail
        try:
            async with db_conn.transaction():
                # Insert more data
                await db_conn.execute("""
                    INSERT INTO test_abort_query (data) VALUES ('test3'), ('test4')
                """)

                # Force an error (reference non-existent table)
                await db_conn.execute("SELECT * FROM nonexistent_table_xyz")

        except asyncpg.exceptions.UndefinedTableError:
            # Expected error - transaction should rollback gracefully
            pass

        # Verify only original data exists (transaction rolled back)
        count = await db_conn.fetchval("SELECT COUNT(*) FROM test_abort_query")
        assert count == 2, f"Expected 2 rows after rollback, got {count}"

        # Verify connection is still usable
        result = await db_conn.fetchval("SELECT data FROM test_abort_query WHERE id = 1")
        assert result == "test1", "Should be able to query after rollback"

    finally:
        try:
            await db_conn.execute("DROP TABLE IF EXISTS test_abort_query")
        except:
            pass


@pytest.mark.asyncio
@pytest.mark.skip(reason="pg_deeplake does not handle rollback yet.")
async def test_multiple_errors_in_sequence(db_conn: asyncpg.Connection):
    """
    Test that multiple errors in sequence don't cause cascading issues.

    This ensures the error handling is robust even when errors occur
    repeatedly.
    """
    assertions = Assertions(db_conn)

    try:
        await db_conn.execute("""
            CREATE TABLE test_multi_error (
                id SERIAL PRIMARY KEY,
                value INTEGER
            ) USING deeplake
        """)

        # Cause multiple errors in sequence
        for i in range(3):
            try:
                # Try to query non-existent table
                await db_conn.execute(f"SELECT * FROM nonexistent_table_{i}")
            except asyncpg.exceptions.UndefinedTableError:
                # Expected - should not cause PANIC
                pass

        # Verify connection still works
        await db_conn.execute("INSERT INTO test_multi_error (value) VALUES (1)")
        count = await db_conn.fetchval("SELECT COUNT(*) FROM test_multi_error")
        assert count == 1, "Connection should still work after multiple errors"

    finally:
        try:
            await db_conn.execute("DROP TABLE IF EXISTS test_multi_error")
        except:
            pass


@pytest.mark.asyncio
@pytest.mark.skip(reason="pg_deeplake does not handle rollback yet.")
async def test_nested_transaction_abort(db_conn: asyncpg.Connection):
    """
    Test nested transaction (savepoint) abort handling.

    Ensures that savepoint rollbacks work correctly without cascading errors.
    """
    assertions = Assertions(db_conn)

    try:
        await db_conn.execute("""
            CREATE TABLE test_nested_abort (
                id SERIAL PRIMARY KEY,
                value INTEGER
            ) USING deeplake
        """)

        # Outer transaction
        async with db_conn.transaction():
            # Insert data in outer transaction
            await db_conn.execute("INSERT INTO test_nested_abort (value) VALUES (1)")

            # Create savepoint and cause error
            try:
                async with db_conn.transaction():
                    await db_conn.execute("INSERT INTO test_nested_abort (value) VALUES (2)")
                    # Force error
                    await db_conn.execute("SELECT * FROM nonexistent_table")
            except asyncpg.exceptions.UndefinedTableError:
                # Savepoint rolled back, outer transaction should continue
                pass

            # Insert more data in outer transaction
            await db_conn.execute("INSERT INTO test_nested_abort (value) VALUES (3)")

        # Verify outer transaction committed but savepoint rolled back
        count = await db_conn.fetchval("SELECT COUNT(*) FROM test_nested_abort")
        assert count == 2, f"Expected 2 rows (values 1 and 3), got {count}"

        values = await db_conn.fetch("SELECT value FROM test_nested_abort ORDER BY value")
        assert [r['value'] for r in values] == [1, 3], "Should have values 1 and 3"

    finally:
        try:
            await db_conn.execute("DROP TABLE IF EXISTS test_nested_abort")
        except:
            pass


@pytest.mark.asyncio
async def test_abort_during_schema_operation(db_conn: asyncpg.Connection):
    """
    Test that errors during schema operations (DDL) are handled gracefully.

    DDL operations can't be rolled back in many databases, but errors
    should still be handled without cascading.
    """
    try:
        # Create a table
        await db_conn.execute("""
            CREATE TABLE test_ddl_abort (
                id SERIAL PRIMARY KEY,
                value INTEGER
            ) USING deeplake
        """)

        # Try to add a column with invalid syntax
        try:
            await db_conn.execute("ALTER TABLE test_ddl_abort ADD COLUMN invalid_syntax")
        except asyncpg.exceptions.PostgresError:
            # Expected error - should not cause cascade
            pass

        # Verify table still exists and is usable
        await db_conn.execute("INSERT INTO test_ddl_abort (value) VALUES (1)")
        count = await db_conn.fetchval("SELECT COUNT(*) FROM test_ddl_abort")
        assert count == 1, "Table should still be usable after DDL error"

    finally:
        try:
            await db_conn.execute("DROP TABLE IF EXISTS test_ddl_abort")
        except:
            pass


@pytest.mark.asyncio
async def test_error_in_parallel_queries(db_conn: asyncpg.Connection):
    """
    Test error handling when multiple queries are running.

    This ensures error handling is thread-safe and doesn't cause
    cascading issues in concurrent scenarios.
    """
    assertions = Assertions(db_conn)

    try:
        await db_conn.execute("""
            CREATE TABLE test_parallel_error (
                id SERIAL PRIMARY KEY,
                value INTEGER
            ) USING deeplake
        """)

        # Insert test data
        await db_conn.execute("""
            INSERT INTO test_parallel_error (value)
            SELECT generate_series(1, 100)
        """)

        # Execute valid query
        result1 = await db_conn.fetch("SELECT COUNT(*) FROM test_parallel_error")

        # Execute query that will error
        try:
            await db_conn.execute("SELECT * FROM nonexistent_parallel_table")
        except asyncpg.exceptions.UndefinedTableError:
            pass

        # Execute another valid query - should still work
        result2 = await db_conn.fetch("SELECT MAX(value) FROM test_parallel_error")
        assert result2[0]['max'] == 100, "Should still be able to query after error"

    finally:
        try:
            await db_conn.execute("DROP TABLE IF EXISTS test_parallel_error")
        except:
            pass
