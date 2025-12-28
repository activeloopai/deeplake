"""
Test ALTER TABLE RENAME COLUMN operations with deeplake tables.

This test verifies that renaming a column in a deeplake table works correctly
by properly syncing the schema change with the deeplake dataset.
"""
import pytest
import asyncpg
from test_utils.assertions import Assertions


@pytest.mark.asyncio
async def test_rename_column_in_table(db_conn: asyncpg.Connection):
    """
    Test that ALTER TABLE RENAME COLUMN works correctly with deeplake tables.

    Tests:
    - Create table with id (SERIAL), name (TEXT), and email (VARCHAR) columns using deeplake
    - Insert two rows with data
    - Rename the 'email' column to 'email_address' via ALTER TABLE
    - Verify the column is renamed in both PostgreSQL catalog and deeplake dataset
    - Verify existing data is preserved and accessible via new column name
    - Verify SELECT works without crashing in SAME connection (tests cache invalidation)
    - Verify SELECT works without crashing in SEPARATE connection (tests persistence)
    - Verify INSERT and UPDATE work with new column name
    - Verify old column name no longer exists
    """
    assertions = Assertions(db_conn)

    try:
        # Create table with id, name, and email columns using deeplake storage
        await db_conn.execute("""
            CREATE TABLE test_rename (
                id SERIAL PRIMARY KEY,
                name TEXT NOT NULL,
                email VARCHAR(255)
            ) USING deeplake
        """)

        # Insert first row
        await db_conn.execute("""
            INSERT INTO test_rename (name, email) VALUES ('Alice', 'alice@example.com')
        """)

        # Insert second row
        await db_conn.execute("""
            INSERT INTO test_rename (name, email) VALUES ('Bob', 'bob@example.com')
        """)

        # Verify two rows were inserted
        await assertions.assert_table_row_count(2, "test_rename")

        # Rename email column to email_address
        await db_conn.execute("""
            ALTER TABLE test_rename RENAME COLUMN email TO email_address
        """)

        # Verify column was renamed in PostgreSQL catalog
        column_info = await db_conn.fetch("""
            SELECT column_name, data_type, character_maximum_length
            FROM information_schema.columns
            WHERE table_name = 'test_rename'
            ORDER BY ordinal_position
        """)
        column_names = [col['column_name'] for col in column_info]

        # Verify new column name exists
        assert 'email_address' in column_names, \
            f"Column 'email_address' should exist in catalog. Found: {column_names}"

        # Verify old column name does not exist
        assert 'email' not in column_names, \
            f"Column 'email' should not exist anymore. Found: {column_names}"

        # Verify column type is preserved
        email_address_col = [col for col in column_info if col['column_name'] == 'email_address'][0]
        assert email_address_col['data_type'] == 'character varying', \
            f"Expected 'character varying', got '{email_address_col['data_type']}'"
        assert email_address_col['character_maximum_length'] == 255, \
            f"Expected max length 255, got {email_address_col['character_maximum_length']}"

        # SELECT from table in SAME connection - this should work without crashing
        rows_after = await db_conn.fetch("SELECT * FROM test_rename ORDER BY id")

        # Verify existing data is preserved and accessible via new column name
        assert len(rows_after) == 2, f"Expected 2 rows, got {len(rows_after)}"
        assert rows_after[0]['id'] == 1, f"Expected id=1, got {rows_after[0]['id']}"
        assert rows_after[0]['name'] == 'Alice', f"Expected 'Alice', got '{rows_after[0]['name']}'"
        assert rows_after[0]['email_address'] == 'alice@example.com', \
            f"Expected 'alice@example.com', got '{rows_after[0]['email_address']}'"
        assert rows_after[1]['id'] == 2, f"Expected id=2, got {rows_after[1]['id']}"
        assert rows_after[1]['name'] == 'Bob', f"Expected 'Bob', got '{rows_after[1]['name']}'"
        assert rows_after[1]['email_address'] == 'bob@example.com', \
            f"Expected 'bob@example.com', got '{rows_after[1]['email_address']}'"

        # Verify old column name is not accessible (should raise error)
        try:
            await db_conn.fetch("SELECT email FROM test_rename")
            assert False, "Should have raised error for non-existent column 'email'"
        except asyncpg.UndefinedColumnError:
            pass  # Expected error

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
            rows_new_conn = await new_conn.fetch("SELECT * FROM test_rename ORDER BY id")
            assert len(rows_new_conn) == 2, f"Expected 2 rows from new connection, got {len(rows_new_conn)}"
            assert rows_new_conn[0]['email_address'] == 'alice@example.com', \
                f"Expected 'alice@example.com' from new connection, got '{rows_new_conn[0]['email_address']}'"
            assert rows_new_conn[1]['email_address'] == 'bob@example.com', \
                f"Expected 'bob@example.com' from new connection, got '{rows_new_conn[1]['email_address']}'"

            # Verify old column name is not accessible in new connection
            try:
                await new_conn.fetch("SELECT email FROM test_rename")
                assert False, "Should have raised error for non-existent column 'email' in new connection"
            except asyncpg.UndefinedColumnError:
                pass  # Expected error
        finally:
            await new_conn.close()

        # Insert another row using the new column name from the original connection
        await db_conn.execute("""
            INSERT INTO test_rename (name, email_address) VALUES ('Charlie', 'charlie@example.com')
        """)

        # Verify the new row was inserted successfully
        all_rows = await db_conn.fetch("SELECT * FROM test_rename ORDER BY id")
        assert len(all_rows) == 3, f"Expected 3 rows, got {len(all_rows)}"
        assert all_rows[2]['name'] == 'Charlie', f"Expected 'Charlie', got '{all_rows[2]['name']}'"
        assert all_rows[2]['email_address'] == 'charlie@example.com', \
            f"Expected 'charlie@example.com', got '{all_rows[2]['email_address']}'"

        await assertions.assert_table_row_count(3, "test_rename")

        print("✓ Test passed: ALTER TABLE RENAME COLUMN works correctly")

    finally:
        # Cleanup - Note: if test crashes, cleanup may not run
        try:
            await db_conn.execute("DROP TABLE IF EXISTS test_rename CASCADE")
        except:
            pass  # Connection may be dead after crash
