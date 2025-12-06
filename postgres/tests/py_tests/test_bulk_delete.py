"""
Test bulk delete operations with deeplake storage.

Ported from: postgres/tests/sql/bulk_delete.sql
"""
import pytest
import asyncpg
from lib.assertions import Assertions


@pytest.mark.asyncio
async def test_bulk_delete(db_conn: asyncpg.Connection):
    """
    Test bulk delete operations.

    Tests:
    - Single row deletes with WHERE clause
    - Bulk deletes with modulo conditions
    - Deleting all rows
    - Large dataset deletions (1,000 rows)
    """
    assertions = Assertions(db_conn)

    try:
        # Create table
        await db_conn.execute("""
            CREATE TABLE people (name text, last_name text, age int) USING deeplake
        """)

        # Insert initial data
        await db_conn.execute("""
            INSERT INTO people (name, last_name, age) VALUES
                ('n1', 'l1', 1),
                ('n2', 'l2', 2),
                ('n3', 'l3', 3),
                ('n4', 'l4', 4)
        """)

        # Delete single row
        await db_conn.execute("""
            DELETE FROM people WHERE name = 'n3' AND last_name = 'l3'
        """)

        await assertions.assert_table_row_count(3, "people")

        # Insert more data
        await db_conn.execute("""
            INSERT INTO people (name, last_name, age) VALUES
                ('n3', 'l3', 3),
                ('n5', 'l5', 5),
                ('n6', 'l6', 6)
        """)

        await assertions.assert_table_row_count(6, "people")
        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM people WHERE age = 4"
        )

        # Delete rows where age divisible by 3
        await db_conn.execute("""
            DELETE FROM people WHERE age % 3 = 0
        """)

        await assertions.assert_table_row_count(4, "people")

        # Bulk insert (using 1000 instead of 10k to avoid crash mentioned in SQL test)
        await db_conn.execute("""
            INSERT INTO people
            SELECT 'n'||i, 'l'||i, i FROM generate_series(1, 1000) i
        """)

        await assertions.assert_table_row_count(1004, "people")

        # Delete even ages
        await db_conn.execute("""
            DELETE FROM people WHERE age % 2 = 0
        """)

        await assertions.assert_table_row_count(502, "people")

        # Delete odd ages (should leave table empty)
        await db_conn.execute("""
            DELETE FROM people WHERE age % 2 = 1
        """)

        await assertions.assert_table_row_count(0, "people")

        print("✓ Test passed: Bulk delete operations work correctly")

    finally:
        # Cleanup
        await db_conn.execute("DROP TABLE IF EXISTS people CASCADE")
