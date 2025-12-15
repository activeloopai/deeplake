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
