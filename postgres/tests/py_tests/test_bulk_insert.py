"""
Test bulk insert operations with deeplake storage.

Ported from: postgres/tests/sql/bulk_insert.sql
"""
import pytest
import asyncpg
from test_utils.assertions import Assertions


@pytest.mark.asyncio
async def test_bulk_insert(db_conn: asyncpg.Connection):
    """
    Test bulk insert operations.

    Tests:
    - Single row inserts
    - Multi-value inserts
    - COPY FROM (using execute_many as equivalent)
    - generate_series bulk inserts (10,000 rows)
    """
    assertions = Assertions(db_conn)

    try:
        # Create table
        await db_conn.execute("""
            CREATE TABLE people (name text, last_name text, age int) USING deeplake
        """)

        # Single row inserts
        await db_conn.execute("""
            INSERT INTO people (name, last_name, age) VALUES ('n1', 'l1', 1)
        """)
        await db_conn.execute("""
            INSERT INTO people (name, last_name, age) VALUES ('n2', 'l2', 2)
        """)
        await db_conn.execute("""
            INSERT INTO people (name, last_name, age) VALUES ('n3', 'l3', 3)
        """)
        await db_conn.execute("""
            INSERT INTO people (name, last_name, age) VALUES ('n4', 'l4', 4)
        """)

        await assertions.assert_table_row_count(4, "people")

        # Multi-value insert
        await db_conn.execute("""
            INSERT INTO people (name, last_name, age) VALUES
                ('n4', 'l4', 4),
                ('n5', 'l5', 5),
                ('n6', 'l6', 6)
        """)

        await assertions.assert_table_row_count(7, "people")
        await assertions.assert_query_row_count(
            2,
            "SELECT * FROM people WHERE age = 4"
        )

        # COPY-equivalent using executemany (Python equivalent of COPY FROM STDIN)
        copy_data = [
            ('n7', 'l7', 7),
            ('n8', 'l8', 8),
            ('n9', 'l9', 9),
        ]
        await db_conn.executemany(
            "INSERT INTO people (name, last_name, age) VALUES ($1, $2, $3)",
            copy_data
        )

        await assertions.assert_table_row_count(10, "people")

        # Bulk insert with generate_series
        await db_conn.execute("""
            INSERT INTO people
            SELECT 'n'||i, 'l'||i, i FROM generate_series(1, 10000) i
        """)

        await assertions.assert_table_row_count(10010, "people")

        print("✓ Test passed: Bulk insert operations work correctly")

    finally:
        # Cleanup
        await db_conn.execute("DROP TABLE IF EXISTS people CASCADE")
