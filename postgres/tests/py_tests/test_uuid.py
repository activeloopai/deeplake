"""
Test UUID data type with deeplake storage.

Ported from: postgres/tests/sql/uuid.sql
"""
import pytest
import asyncpg
from test_utils.assertions import Assertions


@pytest.mark.asyncio
async def test_uuid_type(db_conn: asyncpg.Connection):
    """
    Test UUID data type with deeplake storage.

    Tests:
    - UUID as PRIMARY KEY
    - Default UUID generation (gen_random_uuid())
    - Explicit UUID insertion
    - Exact UUID matching in WHERE clause
    - UUID uniqueness verification
    """
    assertions = Assertions(db_conn)

    try:
        # Create table with UUID primary key
        await db_conn.execute("""
            CREATE TABLE people (
                id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
                name text,
                last_name text,
                age int
            ) USING deeplake
        """)

        # Insert rows with auto-generated UUIDs
        await db_conn.execute("""
            INSERT INTO people(name, last_name, age)
            SELECT 'name' || i, 'last' || i, i
            FROM generate_series(1, 5) AS s(i)
        """)

        # Insert row with explicit UUID
        await db_conn.execute("""
            INSERT INTO people(id, name, last_name, age)
            VALUES ('550e8400-e29b-41d4-a716-446655440000'::uuid, 'John', 'Doe', 30)
        """)

        # Verify total row count
        await assertions.assert_query_row_count(
            6,
            "SELECT * FROM people"
        )

        # Test exact UUID matching
        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM people WHERE id = '550e8400-e29b-41d4-a716-446655440000'"
        )

        # Verify UUID uniqueness
        # Count should match unique IDs
        result = await db_conn.fetchrow("""
            SELECT COUNT(*) AS total_rows, COUNT(DISTINCT id) AS unique_ids
            FROM people
        """)

        assert result['total_rows'] == result['unique_ids'], \
            f"All UUIDs should be unique: {result['total_rows']} rows, {result['unique_ids']} unique IDs"

        # Verify we got exactly 1 row from the uniqueness check query
        await assertions.assert_query_row_count(
            1,
            "SELECT COUNT(*) AS total_rows, COUNT(DISTINCT id) AS unique_ids FROM people"
        )

        print("✓ Test passed: UUID type works correctly")

    finally:
        # Cleanup
        await db_conn.execute("DROP TABLE IF EXISTS people CASCADE")


@pytest.mark.asyncio
async def test_uuid_empty_string_handling(db_conn: asyncpg.Connection):
    """
    Test that empty strings in UUID columns are handled correctly.

    This test verifies the fix for the issue where adding a UUID column
    to a table with existing rows would cause "Failed to parse UUID string:"
    error when querying. Empty strings stored in deeplake are treated as NULL.

    Tests the exact scenario:
    1. Create table without UUID column
    2. Add UUID column (existing rows get NULL/empty values)
    3. Query table (should not crash)
    4. Insert row with NULL UUID
    5. Query multiple times
    """
    assertions = Assertions(db_conn)

    try:
        # Request 1: Create table uuid_test
        await db_conn.execute("""
            CREATE TABLE uuid_test (
                id SERIAL PRIMARY KEY,
                name TEXT NOT NULL
            ) USING deeplake
        """)

        # Request 5: Query on uuid_test table
        rows = await db_conn.fetch(
            "SELECT * FROM (SELECT * FROM uuid_test ORDER BY id) LIMIT 20 OFFSET 0"
        )
        assert len(rows) == 0, f"Expected 0 rows, got {len(rows)}"

        # Request 6: Add UUID column to uuid_test
        await db_conn.execute("""
            ALTER TABLE uuid_test ADD COLUMN uu UUID
        """)

        # Request 8: Query uuid_test after schema change
        # This would previously crash with: ERROR: Failed to parse UUID string:
        rows = await db_conn.fetch(
            "SELECT * FROM (SELECT * FROM uuid_test ORDER BY id) LIMIT 20 OFFSET 0"
        )
        assert len(rows) == 0, f"Expected 0 rows after adding UUID column, got {len(rows)}"

        # Request 9: Insert row with empty name and NULL UUID
        await db_conn.execute("""
            INSERT INTO uuid_test (name, uu) VALUES ('', NULL)
        """)

        # Request 11: Query uuid_test after insert
        rows = await db_conn.fetch(
            "SELECT * FROM (SELECT * FROM uuid_test ORDER BY id) LIMIT 20 OFFSET 0"
        )
        assert len(rows) == 1, f"Expected 1 row after insert, got {len(rows)}"
        assert rows[0]['uu'] is None, "UUID should be NULL"

        # Request 12: Query uuid_test again
        rows = await db_conn.fetch(
            "SELECT * FROM (SELECT * FROM uuid_test ORDER BY id) LIMIT 20 OFFSET 0"
        )
        assert len(rows) == 1, f"Expected 1 row, got {len(rows)}"

        # Request 13: Query uuid_test again
        rows = await db_conn.fetch(
            "SELECT * FROM (SELECT * FROM uuid_test ORDER BY id) LIMIT 20 OFFSET 0"
        )
        assert len(rows) == 1, f"Expected 1 row, got {len(rows)}"

        print("✓ Test passed: UUID empty string handling works correctly")

    finally:
        # Cleanup
        await db_conn.execute("DROP TABLE IF EXISTS uuid_test CASCADE")
