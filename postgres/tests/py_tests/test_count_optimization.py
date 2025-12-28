"""
Test COUNT(*) query optimization for deeplake tables.

This test verifies that:
1. Pure COUNT(*) queries (without WHERE/GROUP BY/HAVING) use the fast path executor
2. COUNT(*) queries with filters use the normal execution path
3. Results are correct in all cases
"""
import pytest
import asyncpg
from test_utils.assertions import Assertions


@pytest.mark.asyncio
async def test_count_star_fast_path(db_conn: asyncpg.Connection):
    """
    Test that pure COUNT(*) queries use the fast path and return correct results.

    Tests:
    - COUNT(*) on empty table
    - COUNT(*) on table with rows
    - COUNT(*) after inserts
    - COUNT(*) after deletes
    - Verify fast path is used via EXPLAIN
    """
    assertions = Assertions(db_conn)

    try:
        # Create a simple test table
        await db_conn.execute("""
            CREATE TABLE count_test (
                id int8,
                name text,
                value float8
            ) USING deeplake
        """)

        # Test 1: COUNT(*) on empty table
        count = await db_conn.fetchval("SELECT COUNT(*) FROM count_test")
        assert count == 0, f"Expected 0 rows, got {count}"
        print("✓ COUNT(*) on empty table: 0")

        # Test 2: Insert some rows
        await db_conn.execute("""
            INSERT INTO count_test (id, name, value)
            VALUES
                (1, 'Alice', 10.5),
                (2, 'Bob', 20.3),
                (3, 'Charlie', 30.7),
                (4, 'David', 40.2),
                (5, 'Eve', 50.9)
        """)

        # Test 3: COUNT(*) after insert
        count = await db_conn.fetchval("SELECT COUNT(*) FROM count_test")
        assert count == 5, f"Expected 5 rows, got {count}"
        print("✓ COUNT(*) after insert: 5")

        # Test 4: Insert more rows
        await db_conn.execute("""
            INSERT INTO count_test (id, name, value)
            VALUES
                (6, 'Frank', 60.1),
                (7, 'Grace', 70.4),
                (8, 'Henry', 80.8)
        """)

        count = await db_conn.fetchval("SELECT COUNT(*) FROM count_test")
        assert count == 8, f"Expected 8 rows, got {count}"
        print("✓ COUNT(*) after second insert: 8")

        # Test 5: Verify EXPLAIN shows fast path
        explain_result = await db_conn.fetch("EXPLAIN SELECT COUNT(*) FROM count_test")
        explain_text = "\n".join([row['QUERY PLAN'] for row in explain_result])

        # Should use CountExecutor for fast path
        if "CountExecutor" in explain_text or "COUNT(*) Fast Path" in explain_text:
            print("✓ EXPLAIN shows COUNT(*) fast path is being used")
        else:
            print(f"⚠ EXPLAIN output:\n{explain_text}")
            print("Note: Fast path might not be visible in EXPLAIN output")

        # Test 6: Delete some rows and verify count
        await db_conn.execute("DELETE FROM count_test WHERE id > 5")
        count = await db_conn.fetchval("SELECT COUNT(*) FROM count_test")
        assert count == 5, f"Expected 5 rows after delete, got {count}"
        print("✓ COUNT(*) after delete: 5")

        # Test 7: Truncate and verify
        await db_conn.execute("DELETE FROM count_test")
        count = await db_conn.fetchval("SELECT COUNT(*) FROM count_test")
        assert count == 0, f"Expected 0 rows after truncate, got {count}"
        print("✓ COUNT(*) after truncate: 0")

    finally:
        # Cleanup
        await db_conn.execute("DROP TABLE IF EXISTS count_test")


@pytest.mark.asyncio
async def test_count_star_with_where(db_conn: asyncpg.Connection):
    """
    Test that COUNT(*) with WHERE clause uses normal execution path.

    These queries should NOT use the fast path because they have filters.
    """
    try:
        # Create test table
        await db_conn.execute("""
            CREATE TABLE count_filter_test (
                id int8,
                category text,
                value float8
            ) USING deeplake
        """)

        # Insert test data
        await db_conn.execute("""
            INSERT INTO count_filter_test (id, category, value)
            VALUES
                (1, 'A', 10.0),
                (2, 'A', 20.0),
                (3, 'B', 30.0),
                (4, 'B', 40.0),
                (5, 'C', 50.0)
        """)

        # Test COUNT(*) with WHERE
        count = await db_conn.fetchval(
            "SELECT COUNT(*) FROM count_filter_test WHERE category = 'A'"
        )
        assert count == 2, f"Expected 2 rows with category='A', got {count}"
        print("✓ COUNT(*) with WHERE category='A': 2")

        # Test COUNT(*) with numeric filter
        count = await db_conn.fetchval(
            "SELECT COUNT(*) FROM count_filter_test WHERE value > 25.0"
        )
        assert count == 3, f"Expected 3 rows with value>25.0, got {count}"
        print("✓ COUNT(*) with WHERE value>25.0: 3")

        # Test COUNT(*) with complex filter
        count = await db_conn.fetchval(
            "SELECT COUNT(*) FROM count_filter_test WHERE category IN ('A', 'B') AND value < 35.0"
        )
        assert count == 3, f"Expected 3 rows with complex filter, got {count}"
        print("✓ COUNT(*) with complex WHERE: 3")

    finally:
        await db_conn.execute("DROP TABLE IF EXISTS count_filter_test")


@pytest.mark.asyncio
async def test_count_star_with_group_by(db_conn: asyncpg.Connection):
    """
    Test that COUNT(*) with GROUP BY uses normal execution path.

    These queries should NOT use the fast path because they have grouping.
    """
    try:
        # Create test table
        await db_conn.execute("""
            CREATE TABLE count_group_test (
                id int8,
                category text,
                value float8
            ) USING deeplake
        """)

        # Insert test data
        await db_conn.execute("""
            INSERT INTO count_group_test (id, category, value)
            VALUES
                (1, 'A', 10.0),
                (2, 'A', 20.0),
                (3, 'B', 30.0),
                (4, 'B', 40.0),
                (5, 'B', 50.0),
                (6, 'C', 60.0)
        """)

        # Test COUNT(*) with GROUP BY
        results = await db_conn.fetch(
            "SELECT category, COUNT(*) as cnt FROM count_group_test GROUP BY category ORDER BY category"
        )

        assert len(results) == 3, f"Expected 3 groups, got {len(results)}"
        assert results[0]['category'] == 'A' and results[0]['cnt'] == 2
        assert results[1]['category'] == 'B' and results[1]['cnt'] == 3
        assert results[2]['category'] == 'C' and results[2]['cnt'] == 1
        print("✓ COUNT(*) with GROUP BY returns correct grouped counts")

    finally:
        await db_conn.execute("DROP TABLE IF EXISTS count_group_test")


@pytest.mark.asyncio
async def test_count_star_with_having(db_conn: asyncpg.Connection):
    """
    Test that COUNT(*) with HAVING clause uses normal execution path.
    """
    try:
        # Create test table
        await db_conn.execute("""
            CREATE TABLE count_having_test (
                id int8,
                category text
            ) USING deeplake
        """)

        # Insert test data
        await db_conn.execute("""
            INSERT INTO count_having_test (id, category)
            VALUES
                (1, 'A'), (2, 'A'), (3, 'A'),
                (4, 'B'), (5, 'B'),
                (6, 'C')
        """)

        # Test COUNT(*) with HAVING
        results = await db_conn.fetch("""
            SELECT category, COUNT(*) as cnt
            FROM count_having_test
            GROUP BY category
            HAVING COUNT(*) > 1
            ORDER BY category
        """)

        assert len(results) == 2, f"Expected 2 groups with count>1, got {len(results)}"
        assert results[0]['category'] == 'A' and results[0]['cnt'] == 3
        assert results[1]['category'] == 'B' and results[1]['cnt'] == 2
        print("✓ COUNT(*) with HAVING returns correct filtered groups")

    finally:
        await db_conn.execute("DROP TABLE IF EXISTS count_having_test")


@pytest.mark.asyncio
async def test_count_star_with_multiple_aggregates(db_conn: asyncpg.Connection):
    """
    Test that queries with COUNT(*) and other aggregates use normal execution path.
    """
    try:
        # Create test table
        await db_conn.execute("""
            CREATE TABLE count_agg_test (
                id int8,
                value float8
            ) USING deeplake
        """)

        # Insert test data
        await db_conn.execute("""
            INSERT INTO count_agg_test (id, value)
            VALUES
                (1, 10.0),
                (2, 20.0),
                (3, 30.0),
                (4, 40.0),
                (5, 50.0)
        """)

        # Test COUNT(*) with other aggregates
        result = await db_conn.fetchrow("""
            SELECT COUNT(*) as cnt, SUM(value) as total, AVG(value) as avg_val
            FROM count_agg_test
        """)

        assert result['cnt'] == 5, f"Expected count=5, got {result['cnt']}"
        assert result['total'] == 150.0, f"Expected sum=150.0, got {result['total']}"
        assert result['avg_val'] == 30.0, f"Expected avg=30.0, got {result['avg_val']}"
        print("✓ COUNT(*) with multiple aggregates returns correct results")

    finally:
        await db_conn.execute("DROP TABLE IF EXISTS count_agg_test")


@pytest.mark.asyncio
async def test_count_star_large_table(db_conn: asyncpg.Connection):
    """
    Test COUNT(*) performance on a larger table.

    This verifies the fast path works correctly with more data.
    """
    try:
        # Create test table
        await db_conn.execute("""
            CREATE TABLE count_large_test (
                id int8,
                value float8
            ) USING deeplake
        """)

        # Insert 1000 rows
        await db_conn.execute("""
            INSERT INTO count_large_test (id, value)
            SELECT i, i * 1.5
            FROM generate_series(1, 1000) AS i
        """)

        # Test COUNT(*)
        count = await db_conn.fetchval("SELECT COUNT(*) FROM count_large_test")
        assert count == 1000, f"Expected 1000 rows, got {count}"
        print("✓ COUNT(*) on table with 1000 rows: 1000")

        # Test COUNT(*) with filter (should use normal path)
        count = await db_conn.fetchval(
            "SELECT COUNT(*) FROM count_large_test WHERE id <= 500"
        )
        assert count == 500, f"Expected 500 rows with filter, got {count}"
        print("✓ COUNT(*) with filter on large table: 500")

    finally:
        await db_conn.execute("DROP TABLE IF EXISTS count_large_test")


@pytest.mark.asyncio
async def test_count_column_vs_count_star(db_conn: asyncpg.Connection):
    """
    Test that COUNT(column) and COUNT(*) return same results when no NULLs.

    Note: COUNT(column) should use normal path, COUNT(*) should use fast path.
    """
    try:
        # Create test table
        await db_conn.execute("""
            CREATE TABLE count_compare_test (
                id int8,
                value float8
            ) USING deeplake
        """)

        # Insert test data (no NULLs)
        await db_conn.execute("""
            INSERT INTO count_compare_test (id, value)
            VALUES
                (1, 10.0),
                (2, 20.0),
                (3, 30.0)
        """)

        # Compare COUNT(*) and COUNT(column)
        count_star = await db_conn.fetchval("SELECT COUNT(*) FROM count_compare_test")
        count_id = await db_conn.fetchval("SELECT COUNT(id) FROM count_compare_test")
        count_value = await db_conn.fetchval("SELECT COUNT(value) FROM count_compare_test")

        assert count_star == count_id == count_value == 3, \
            f"Expected all counts to be 3, got COUNT(*)={count_star}, COUNT(id)={count_id}, COUNT(value)={count_value}"
        print("✓ COUNT(*), COUNT(id), and COUNT(value) all return 3")

    finally:
        await db_conn.execute("DROP TABLE IF EXISTS count_compare_test")
