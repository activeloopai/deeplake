"""
Test ALTER TABLE ADD COLUMN operations with deeplake tables.

This test verifies that adding a column to a deeplake table works correctly
after the fix that properly syncs the schema with the deeplake dataset.
"""
import pytest
import asyncpg
from test_utils.assertions import Assertions


@pytest.mark.asyncio
async def test_add_column_to_table(db_conn: asyncpg.Connection):
    """
    Test that ALTER TABLE ADD COLUMN works correctly with deeplake tables.

    Tests:
    - Create table with id (SERIAL) and name (TEXT) columns using deeplake
    - Insert two rows with separate INSERT queries
    - Add a new varchar(255) column via ALTER TABLE
    - Verify the column is added to both PostgreSQL catalog and deeplake dataset
    - Verify existing data is preserved and new column contains NULL for existing rows
    - Verify SELECT works without crashing in SAME connection (tests cache invalidation)
    - Verify SELECT works without crashing in SEPARATE connection (tests persistence)
    - Verify INSERT works with new column from both connections
    """
    assertions = Assertions(db_conn)

    try:
        # Create table with id and text columns using deeplake storage
        await db_conn.execute("""
            CREATE TABLE test_crash (
                id SERIAL PRIMARY KEY,
                name TEXT NOT NULL
            ) USING deeplake
        """)

        # Insert first row
        await db_conn.execute("""
            INSERT INTO test_crash (name) VALUES ('')
        """)

        # Insert second row
        await db_conn.execute("""
            INSERT INTO test_crash (name) VALUES ('')
        """)

        # Verify two rows were inserted
        await assertions.assert_table_row_count(2, "test_crash")

        # Add new column with varchar(255) type
        # This ALTER TABLE succeeds...
        await db_conn.execute("""
            ALTER TABLE test_crash ADD COLUMN valod VARCHAR(255)
        """)

        # Verify column was added to PostgreSQL catalog
        column_info = await db_conn.fetch("""
            SELECT column_name, data_type, character_maximum_length
            FROM information_schema.columns
            WHERE table_name = 'test_crash'
            ORDER BY ordinal_position
        """)
        column_names = [col['column_name'] for col in column_info]
        assert 'valod' in column_names, \
            f"Column 'valod' should exist in catalog. Found: {column_names}"

        # Verify column type
        valod_col = [col for col in column_info if col['column_name'] == 'valod'][0]
        assert valod_col['data_type'] == 'character varying', \
            f"Expected 'character varying', got '{valod_col['data_type']}'"
        assert valod_col['character_maximum_length'] == 255, \
            f"Expected max length 255, got {valod_col['character_maximum_length']}"

        # SELECT from table in SAME connection - this should work without crashing
        rows_after = await db_conn.fetch("SELECT * FROM test_crash ORDER BY id")

        # Verify existing data is preserved
        # Note: deeplake returns empty string for text fields instead of NULL
        assert len(rows_after) == 2, f"Expected 2 rows, got {len(rows_after)}"
        assert rows_after[0]['id'] == 1, f"Expected id=1, got {rows_after[0]['id']}"
        assert rows_after[0]['name'] == '', f"Expected empty string, got '{rows_after[0]['name']}'"
        assert rows_after[0]['valod'] == '', \
            f"Expected empty string for new column, got '{rows_after[0]['valod']}'"
        assert rows_after[1]['id'] == 2, f"Expected id=2, got {rows_after[1]['id']}"
        assert rows_after[1]['name'] == '', f"Expected empty string, got '{rows_after[1]['name']}'"
        assert rows_after[1]['valod'] == '', \
            f"Expected empty string for new column, got '{rows_after[1]['valod']}'"

        # Test with a SEPARATE connection - should also work
        import os
        user = os.environ.get("USER", "postgres")
        new_conn = await asyncpg.connect(
            database="postgres",
            user=user,
            host="localhost",
            statement_cache_size=0
        )
        try:
            rows_new_conn = await new_conn.fetch("SELECT * FROM test_crash ORDER BY id")
            assert len(rows_new_conn) == 2, f"Expected 2 rows from new connection, got {len(rows_new_conn)}"
            assert rows_new_conn[0]['valod'] == '', \
                f"Expected empty string for new column from new connection, got '{rows_new_conn[0]['valod']}'"
            assert rows_new_conn[1]['valod'] == '', \
                f"Expected empty string for new column from new connection, got '{rows_new_conn[1]['valod']}'"

            # Also verify we can insert from the new connection
            await new_conn.execute("""
                INSERT INTO test_crash (name, valod) VALUES ('from new conn', 'test value')
            """)
            row_from_new_conn = await new_conn.fetchrow("""
                SELECT * FROM test_crash WHERE name = 'from new conn'
            """)
            assert row_from_new_conn is not None, "Expected to find row inserted from new connection"
            assert row_from_new_conn['valod'] == 'test value', \
                f"Expected 'test value', got '{row_from_new_conn['valod']}'"
        finally:
            await new_conn.close()

        # Insert another row with the new column from the original connection
        await db_conn.execute("""
            INSERT INTO test_crash (name, valod) VALUES ('test', 'new value')
        """)

        # Verify the new row
        new_row = await db_conn.fetchrow("""
            SELECT * FROM test_crash WHERE name = 'test'
        """)
        assert new_row is not None, "Expected to find row with name='test'"
        assert new_row['name'] == 'test', f"Expected 'test', got '{new_row['name']}'"
        assert new_row['valod'] == 'new value', \
            f"Expected 'new value', got '{new_row['valod']}'"

        # Verify total row count (2 original + 1 from new connection + 1 from original connection)
        await assertions.assert_table_row_count(4, "test_crash")

        print("✓ Test passed: ALTER TABLE ADD COLUMN works correctly")

    finally:
        # Cleanup - Note: if test crashes, cleanup may not run
        try:
            await db_conn.execute("DROP TABLE IF EXISTS test_crash CASCADE")
        except:
            pass  # Connection may be dead after segfault


@pytest.mark.asyncio
async def test_add_column_with_custom_schema(db_conn: asyncpg.Connection):
    """
    Test that ALTER TABLE ADD COLUMN works correctly with custom schema.

    Tests:
    - Create a custom schema 'valod'
    - Create table with id (INT4) and name (TEXT) columns using deeplake
    - Insert one row with explicit id and empty string
    - Verify data before adding column
    - Add a new TEXT column 'surname' via ALTER TABLE
    - Verify the column is added and existing data is preserved
    - Verify SELECT returns correct columns after ALTER TABLE
    """
    assertions = Assertions(db_conn)

    try:
        # Create custom schema
        await db_conn.execute("CREATE SCHEMA valod")

        # Set search path to use the custom schema
        await db_conn.execute("SET search_path TO 'valod'")

        # Create table with id and name columns using deeplake storage
        await db_conn.execute("""
            CREATE TABLE a (
                id INT4,
                name TEXT
            ) USING deeplake
        """)

        # Insert one row with explicit id and empty string
        await db_conn.execute("INSERT INTO a VALUES (1, '')")

        # Verify row was inserted
        rows_before = await db_conn.fetch("SELECT * FROM a")
        assert len(rows_before) == 1, f"Expected 1 row, got {len(rows_before)}"
        assert rows_before[0]['id'] == 1, f"Expected id=1, got {rows_before[0]['id']}"
        assert rows_before[0]['name'] == '', f"Expected empty string, got '{rows_before[0]['name']}'"

        # Add new column 'surname' with TEXT type
        await db_conn.execute("ALTER TABLE a ADD COLUMN surname TEXT")

        # Verify column was added to PostgreSQL catalog
        column_info = await db_conn.fetch("""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = 'valod' AND table_name = 'a'
            ORDER BY ordinal_position
        """)
        column_names = [col['column_name'] for col in column_info]
        assert column_names == ['id', 'name', 'surname'], \
            f"Expected ['id', 'name', 'surname'], got {column_names}"

        # SELECT from table after adding column
        rows_after = await db_conn.fetch("SELECT * FROM a")

        # Verify existing data is preserved and new column exists
        assert len(rows_after) == 1, f"Expected 1 row, got {len(rows_after)}"
        assert rows_after[0]['id'] == 1, f"Expected id=1, got {rows_after[0]['id']}"
        assert rows_after[0]['name'] == '', f"Expected empty string, got '{rows_after[0]['name']}'"
        assert 'surname' in rows_after[0].keys(), "Expected 'surname' column to exist"
        assert rows_after[0]['surname'] == '', \
            f"Expected empty string for new column, got '{rows_after[0]['surname']}'"

        print("✓ Test passed: ALTER TABLE ADD COLUMN works with custom schema")

    finally:
        # Cleanup
        try:
            await db_conn.execute("DROP SCHEMA IF EXISTS valod CASCADE")
        except:
            pass  # Connection may be dead after segfault
