"""
Test basic operations with deeplake tables and indexes.

Ported from: postgres/tests/sql/basic_ops.sql
"""
import pytest
import asyncpg
from lib.assertions import Assertions


@pytest.mark.asyncio
async def test_basic_vector_operations(db_conn: asyncpg.Connection):
    """
    Test basic vector operations with deeplake index.

    This test:
    - Creates a table with float4[] columns using deeplake storage
    - Creates a deeplake_index on a vector column
    - Inserts test vectors
    - Performs similarity search using <#> operator
    - Validates results match expected scores
    """
    assertions = Assertions(db_conn)

    try:
        # Create table with deeplake storage
        await db_conn.execute("""
            CREATE TABLE vectors (
                id SERIAL PRIMARY KEY,
                v1 float4[],
                v2 float4[]
            ) USING deeplake
        """)

        # Create deeplake index
        await db_conn.execute("""
            CREATE INDEX index_for_v1 ON vectors USING deeplake_index (v1 DESC)
        """)

        # Verify index exists
        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM pg_class WHERE relname = 'index_for_v1'"
        )

        # Insert test vectors
        await db_conn.execute("""
            INSERT INTO vectors (v1, v2) VALUES
                (ARRAY[1.0, 2.0, 3.0], ARRAY[1.0, 2.0, 3.0]),
                (ARRAY[4.0, 5.0, 6.0], ARRAY[4.0, 5.0, 6.0]),
                (ARRAY[7.0, 8.0, 9.0], ARRAY[7.0, 8.0, 9.0]),
                (ARRAY[0.0, 0.0, 444], ARRAY[0.0, 0.0, 444])
        """)

        # Verify row count
        await assertions.assert_table_row_count(4, "vectors")

        # Test index usage - disable sequential scan to force index usage
        await db_conn.execute("SET enable_seqscan = off")

        # Create expected results
        await db_conn.execute("""
            CREATE TEMP TABLE expected_vectors (id INTEGER, score REAL)
        """)
        await db_conn.execute("""
            INSERT INTO expected_vectors VALUES
                (1, 1),
                (2, 0.97463185),
                (3, 0.959412),
                (4, 0.80178374)
        """)

        # Execute similarity search
        results = await db_conn.fetch("""
            SELECT id, v1 <#> ARRAY[1.0, 2.0, 3.0] AS score
            FROM vectors
            ORDER BY score
            LIMIT 5
        """)

        # Validate results
        expected_scores = {
            1: 1.0,
            2: 0.97463185,
            3: 0.959412,
            4: 0.80178374
        }

        assert len(results) == 4, f"Expected 4 results, got {len(results)}"

        for row in results:
            expected_score = expected_scores[row['id']]
            actual_score = row['score']
            assert abs(actual_score - expected_score) < 1e-6, (
                f"Score mismatch for id {row['id']}: "
                f"expected {expected_score}, got {actual_score}"
            )

        print("✓ Test passed: Basic vector operations work correctly")

    finally:
        # Cleanup
        await db_conn.execute("RESET enable_seqscan")
        await db_conn.execute("DROP TABLE IF EXISTS vectors CASCADE")
