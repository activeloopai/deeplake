"""
Test default PostgreSQL index behavior with deeplake table storage.

Ported from: postgres/tests/sql/default_index.sql
"""
import pytest
import asyncpg
from lib.assertions import Assertions


@pytest.mark.asyncio
@pytest.mark.slow
async def test_default_index(db_conn: asyncpg.Connection):
    """
    Test that standard PostgreSQL indexes work with deeplake table storage.

    Tests:
    - Creating deeplake table (USING deeplake)
    - Inserting 200k+ rows
    - Query behavior without index (sequential scan)
    - Creating standard PostgreSQL index (not deeplake_index)
    - Query behavior with index (index scan)
    - Dropping the index
    """
    assertions = Assertions(db_conn)

    try:
        # Disable deeplake executor to use standard PostgreSQL executor
        await db_conn.execute("SET pg_deeplake.use_deeplake_executor = off")

        # Create table using deeplake storage
        await db_conn.execute("""
            CREATE TABLE people (name text, last_name text, age int) USING deeplake
        """)

        # Insert initial rows
        await db_conn.execute("""
            INSERT INTO people (name, last_name, age) VALUES
                ('n1', 'l1', 1),
                ('n2', 'l2', 2),
                ('n3', 'l3', 3),
                ('n4', 'l4', 4)
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

        # Bulk insert 200k rows (100k twice)
        await db_conn.execute("""
            INSERT INTO people
            SELECT 'n'||i, 'l'||i, i FROM generate_series(1, 100000) i
        """)

        await db_conn.execute("""
            INSERT INTO people
            SELECT 'n'||i, 'l'||i, i FROM generate_series(1, 100000) i
        """)

        await assertions.assert_table_row_count(200007, "people")

        # Verify query doesn't use index scan initially (should use Seq Scan)
        explain_result = await db_conn.fetch(
            "EXPLAIN SELECT ctid, age FROM people WHERE age = 4"
        )
        explain_text = "\n".join([row[0] for row in explain_result])

        assert "Index Scan" not in explain_text, \
            f"Query should not use Index Scan without index. Got: {explain_text}"

        # Create standard PostgreSQL index (not deeplake_index)
        await db_conn.execute("CREATE INDEX idx_people_age ON people(age)")

        # Verify query now uses index scan
        explain_result_with_idx = await db_conn.fetch(
            "EXPLAIN SELECT ctid, age FROM people WHERE age = 4"
        )
        explain_text_with_idx = "\n".join([row[0] for row in explain_result_with_idx])

        # Note: Could be "Index Scan" or "Bitmap Index Scan" depending on planner
        has_index_scan = ("Index Scan" in explain_text_with_idx or
                         "Bitmap" in explain_text_with_idx)
        assert has_index_scan, \
            f"Query should use Index Scan with index. Got: {explain_text_with_idx}"

        # Verify result count
        await assertions.assert_query_row_count(
            4,
            "SELECT ctid, age FROM people WHERE age = 4"
        )

        # Drop the index
        await db_conn.execute("DROP INDEX idx_people_age")

        print("✓ Test passed: Default PostgreSQL index works with deeplake table storage")

    finally:
        # Cleanup
        await db_conn.execute("DROP TABLE IF EXISTS people CASCADE")
        await db_conn.execute("RESET pg_deeplake.use_deeplake_executor")
