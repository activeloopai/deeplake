"""
Test for UUID NULL value bug during UPDATE operations.

Bug: When a table has a UUID column with NULL values and another column (e.g., bool),
attempting to update the non-UUID column fails with "invalid value for type uuid".

Reproduction steps:
1. Create table with UUID column and bool column using DEEPLAKE access method
2. Add rows with NULL UUID values
3. Update the bool column value
4. Expected: Update should succeed
5. Actual: Fails with "invalid value for type uuid"
"""
import pytest
import asyncpg
from test_utils.assertions import Assertions


@pytest.mark.asyncio
async def test_uuid_null_update_bug(db_conn: asyncpg.Connection):
    """
    Test that updating non-UUID columns works when UUID column contains NULL values.

    This reproduces the bug where updating a bool column fails when there's a
    UUID column with NULL values in the same table.
    """
    assertions = Assertions(db_conn)

    try:
        # Step 1: Create table with UUID and bool columns using DEEPLAKE access method
        await db_conn.execute("""
            CREATE TABLE test_uuid_bool (
                id SERIAL PRIMARY KEY,
                uuid_col UUID,
                bool_col BOOL
            ) USING deeplake
        """)
        print("✓ Created table with UUID and bool columns")

        # Step 2: Add rows with NULL/empty values
        await db_conn.execute("""
            INSERT INTO test_uuid_bool (uuid_col, bool_col) VALUES
            (NULL, NULL),
            (NULL, FALSE),
            (NULL, TRUE)
        """)
        print("✓ Inserted rows with NULL UUID values")

        # Verify initial state
        rows = await db_conn.fetch("SELECT * FROM test_uuid_bool ORDER BY id")
        assert len(rows) == 3, f"Expected 3 rows, got {len(rows)}"
        assert all(row['uuid_col'] is None for row in rows), "All UUID values should be NULL"
        print(f"✓ Verified initial state: {len(rows)} rows with NULL UUIDs")

        # Step 3: Attempt to update the bool column
        # This is where the bug occurs - it should succeed but fails with
        # "invalid value for type uuid"
        try:
            await db_conn.execute("""
                UPDATE test_uuid_bool
                SET bool_col = TRUE
                WHERE id = 1
            """)
            print("✓ Successfully updated bool column (bug is fixed!)")
        except Exception as e:
            print(f"✗ Failed to update bool column: {e}")
            raise AssertionError(f"Update failed with error: {e}. This is the bug we're testing for.")

        # Verify the update worked
        row = await db_conn.fetchrow("SELECT * FROM test_uuid_bool WHERE id = 1")
        assert row['bool_col'] is True, "bool_col should be TRUE after update"
        assert row['uuid_col'] is None, "uuid_col should still be NULL"
        print("✓ Update succeeded and values are correct")

        # Test additional update scenarios
        await db_conn.execute("""
            UPDATE test_uuid_bool
            SET bool_col = FALSE
            WHERE uuid_col IS NULL
        """)
        print("✓ Batch update with NULL UUID condition succeeded")

        # Verify all bool values are now FALSE
        rows = await db_conn.fetch("SELECT * FROM test_uuid_bool ORDER BY id")
        assert all(row['bool_col'] is False for row in rows), "All bool values should be FALSE"
        print("✓ Batch update worked correctly")

    finally:
        # Cleanup
        await db_conn.execute("DROP TABLE IF EXISTS test_uuid_bool CASCADE")


@pytest.mark.asyncio
async def test_uuid_null_multiple_columns_update(db_conn: asyncpg.Connection):
    """
    Test updating multiple different column types when UUID column has NULL values.

    This is a more comprehensive test covering multiple data types.
    """
    try:
        # Create table with UUID and various other column types
        await db_conn.execute("""
            CREATE TABLE test_uuid_multi (
                id SERIAL PRIMARY KEY,
                uuid_col UUID,
                bool_col BOOL,
                int_col INT,
                text_col TEXT
            ) USING deeplake
        """)
        print("✓ Created table with UUID and multiple column types")

        # Insert rows with NULL UUID values
        await db_conn.execute("""
            INSERT INTO test_uuid_multi (uuid_col, bool_col, int_col, text_col) VALUES
            (NULL, NULL, NULL, NULL),
            (NULL, TRUE, 42, 'test'),
            ('550e8400-e29b-41d4-a716-446655440000'::uuid, FALSE, 100, 'with_uuid')
        """)
        print("✓ Inserted rows with mixed UUID values (NULL and non-NULL)")

        # Test updating bool column where UUID is NULL
        await db_conn.execute("""
            UPDATE test_uuid_multi
            SET bool_col = TRUE
            WHERE id = 1
        """)
        print("✓ Updated bool column for row with NULL UUID")

        # Test updating int column where UUID is NULL
        await db_conn.execute("""
            UPDATE test_uuid_multi
            SET int_col = 999
            WHERE id = 1
        """)
        print("✓ Updated int column for row with NULL UUID")

        # Test updating text column where UUID is NULL
        await db_conn.execute("""
            UPDATE test_uuid_multi
            SET text_col = 'updated'
            WHERE id = 1
        """)
        print("✓ Updated text column for row with NULL UUID")

        # Test updating multiple columns at once
        await db_conn.execute("""
            UPDATE test_uuid_multi
            SET bool_col = FALSE, int_col = 777, text_col = 'multi_update'
            WHERE id = 2
        """)
        print("✓ Updated multiple columns at once for row with NULL UUID")

        # Verify all updates worked correctly
        row = await db_conn.fetchrow("SELECT * FROM test_uuid_multi WHERE id = 1")
        assert row['bool_col'] is True, "bool_col should be TRUE"
        assert row['int_col'] == 999, "int_col should be 999"
        assert row['text_col'] == 'updated', "text_col should be 'updated'"
        assert row['uuid_col'] is None, "uuid_col should still be NULL"

        row = await db_conn.fetchrow("SELECT * FROM test_uuid_multi WHERE id = 2")
        assert row['bool_col'] is False, "bool_col should be FALSE"
        assert row['int_col'] == 777, "int_col should be 777"
        assert row['text_col'] == 'multi_update', "text_col should be 'multi_update'"
        assert row['uuid_col'] is None, "uuid_col should still be NULL"

        print("✓ All updates verified correctly")

    finally:
        # Cleanup
        await db_conn.execute("DROP TABLE IF EXISTS test_uuid_multi CASCADE")
