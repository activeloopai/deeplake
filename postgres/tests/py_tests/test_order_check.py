"""
Test ORDER BY behavior with DESC index and similarity search.

Ported from: postgres/tests/sql/order_check.sql
"""
import pytest
import asyncpg
from test_utils.assertions import Assertions


@pytest.mark.asyncio
async def test_order_check(db_conn: asyncpg.Connection):
    """
    Test that ORDER BY with similarity operator returns correct ordering.

    Tests:
    - Creating DESC deeplake_index
    - Similarity search with ORDER BY ... DESC
    - Similarity search with ORDER BY (ASC implicit)
    - Verifying result ordering
    """
    assertions = Assertions(db_conn)

    try:
        # Create table
        await db_conn.execute("""
            CREATE TABLE vectors (
                id SERIAL PRIMARY KEY,
                v1 float4[],
                v2 float4[]
            ) USING deeplake
        """)

        # Create DESC index on v2
        await db_conn.execute("""
            CREATE INDEX desc_index ON vectors USING deeplake_index (v2 DESC)
        """)

        # Verify index exists
        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM pg_class WHERE relname = 'desc_index'"
        )

        # Insert test vectors
        await db_conn.execute("""
            INSERT INTO vectors (v1, v2) VALUES
                (ARRAY[1.0, 2.0, 3.0], ARRAY[1.0, 2.0, 3.0]),
                (ARRAY[4.0, 5.0, 6.0], ARRAY[4.0, 5.0, 6.0]),
                (ARRAY[7.0, 8.0, 9.0], ARRAY[7.0, 8.0, 9.0]),
                (ARRAY[10.0, 11.0, 12.0], ARRAY[10.0, 11.0, 12.0])
        """)

        # Disable sequential scan to force index usage
        await db_conn.execute("SET enable_seqscan = off")

        # Test 1: ORDER BY v1 <#> ... DESC LIMIT 5
        # Expected order: farthest to closest (DESC)
        # Query point: [4.1, 5.2, 6.3]
        # Expected: (2), (3), (4), (1) - in DESC distance order
        result1 = await db_conn.fetch("""
            SELECT * FROM vectors
            ORDER BY v1 <#> ARRAY[4.1, 5.2, 6.3] DESC
            LIMIT 5
        """)

        assert len(result1) == 4, f"Expected 4 results, got {len(result1)}"

        # Verify ordering: should be DESC by distance (farthest first)
        expected_ids_1 = [2, 3, 4, 1]
        actual_ids_1 = [row['id'] for row in result1]

        assert actual_ids_1 == expected_ids_1, \
            f"Expected order {expected_ids_1}, got {actual_ids_1}"

        # Verify the actual vectors
        assert result1[0]['v1'] == [4.0, 5.0, 6.0], \
            f"Expected first vector [4.0, 5.0, 6.0], got {result1[0]['v1']}"
        assert result1[1]['v1'] == [7.0, 8.0, 9.0], \
            f"Expected second vector [7.0, 8.0, 9.0], got {result1[1]['v1']}"
        assert result1[2]['v1'] == [10.0, 11.0, 12.0], \
            f"Expected third vector [10.0, 11.0, 12.0], got {result1[2]['v1']}"
        assert result1[3]['v1'] == [1.0, 2.0, 3.0], \
            f"Expected fourth vector [1.0, 2.0, 3.0], got {result1[3]['v1']}"

        # Test 2: ORDER BY v2 <#> ... LIMIT 5 (ASC implicit)
        # Expected order: closest to farthest (ASC)
        # Query point: [4.0, 5.0, 6.0]
        # Expected: (2), (3), (4), (1) - in ASC distance order
        result2 = await db_conn.fetch("""
            SELECT * FROM vectors
            ORDER BY v2 <#> ARRAY[4.0, 5.0, 6.0]
            LIMIT 5
        """)

        assert len(result2) == 4, f"Expected 4 results, got {len(result2)}"

        # Verify ordering: should be ASC by distance (closest first)
        expected_ids_2 = [2, 3, 4, 1]
        actual_ids_2 = [row['id'] for row in result2]

        assert actual_ids_2 == expected_ids_2, \
            f"Expected order {expected_ids_2}, got {actual_ids_2}"

        # Verify the actual vectors
        assert result2[0]['v2'] == [4.0, 5.0, 6.0], \
            f"Expected first vector [4.0, 5.0, 6.0], got {result2[0]['v2']}"
        assert result2[1]['v2'] == [7.0, 8.0, 9.0], \
            f"Expected second vector [7.0, 8.0, 9.0], got {result2[1]['v2']}"
        assert result2[2]['v2'] == [10.0, 11.0, 12.0], \
            f"Expected third vector [10.0, 11.0, 12.0], got {result2[2]['v2']}"
        assert result2[3]['v2'] == [1.0, 2.0, 3.0], \
            f"Expected fourth vector [1.0, 2.0, 3.0], got {result2[3]['v2']}"

        print("✓ Test passed: ORDER BY with similarity operator works correctly")

    finally:
        # Cleanup
        await db_conn.execute("DROP INDEX IF EXISTS asc_index CASCADE")
        await db_conn.execute("DROP INDEX IF EXISTS desc_index CASCADE")
        await db_conn.execute("DROP TABLE IF EXISTS vectors CASCADE")
        await db_conn.execute("RESET enable_seqscan")
