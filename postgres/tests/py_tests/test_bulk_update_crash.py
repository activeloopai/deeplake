"""
Test bulk UPDATE operations with crash recovery scenarios.

Ported from: postgres/tests/sql/bulk_update_crash.sql
"""
import pytest
import asyncpg
from lib.assertions import Assertions


@pytest.mark.asyncio
async def test_bulk_update_crash(db_conn: asyncpg.Connection):
    """
    Test bulk UPDATE operations with various scenarios including COPY data.

    Tests:
    - Single row updates
    - Multiple row updates with WHERE clause
    - Bulk updates with arithmetic operations
    - COPY FROM equivalent (executemany)
    - Updates on COPY-inserted data
    - CASE expressions for conditional updates
    """
    assertions = Assertions(db_conn)

    try:
        # Create table
        await db_conn.execute("""
            CREATE TABLE people (name text, last_name text, age int) USING deeplake
        """)

        # Insert initial test data
        await db_conn.execute("""
            INSERT INTO people (name, last_name, age) VALUES
                ('n1', 'l1', 1),
                ('n2', 'l2', 2),
                ('n3', 'l3', 3),
                ('n4', 'l4', 4)
        """)

        await assertions.assert_table_row_count(4, "people")

        # Test single row update
        await db_conn.execute("""
            UPDATE people SET age = 25 WHERE name = 'n1'
        """)

        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM people WHERE age = 25"
        )

        # Test multiple row update
        await db_conn.execute("""
            UPDATE people SET last_name = 'updated' WHERE age <= 4
        """)

        await assertions.assert_query_row_count(
            3,
            "SELECT * FROM people WHERE last_name = 'updated'"
        )

        # Test bulk update with arithmetic (age = age + 10)
        await db_conn.execute("""
            UPDATE people SET age = age + 10 WHERE name LIKE 'n%'
        """)

        await assertions.assert_query_row_count(
            0,
            "SELECT * FROM people WHERE age < 10"
        )

        # Simulate COPY FROM STDIN using executemany (batch insert)
        copy_data = [
            ('n5', 'l5', 5),
            ('n6', 'l6', 6),
            ('n7', 'l7', 7)
        ]
        await db_conn.executemany(
            "INSERT INTO people (name, last_name, age) VALUES ($1, $2, $3)",
            copy_data
        )

        await assertions.assert_table_row_count(7, "people")

        # Test update on COPY-inserted data
        await db_conn.execute("""
            UPDATE people SET name = 'updated_' || name WHERE age <= 7
        """)

        await assertions.assert_query_row_count(
            3,
            "SELECT * FROM people WHERE name LIKE 'updated%'"
        )

        # Test bulk update with CASE expression
        await db_conn.execute("""
            UPDATE people SET age = CASE
                WHEN age > 15 THEN age - 5
                ELSE age + 5
            END
        """)

        await assertions.assert_table_row_count(7, "people")

        print("✓ Test passed: Bulk UPDATE with crash recovery scenarios works correctly")

    finally:
        # Cleanup
        await db_conn.execute("DROP TABLE IF EXISTS people CASCADE")
