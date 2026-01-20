"""
Test UPDATE operations with deeplake storage and indexes.

Ported from: postgres/tests/sql/update_table.sql
"""
import pytest
import asyncpg
from test_utils.assertions import Assertions


@pytest.mark.asyncio
async def test_update_with_index(db_conn: asyncpg.Connection):
    """
    Test UPDATE operations on tables with deeplake indexes.

    Tests:
    - Creating table with vector columns
    - Creating deeplake_index
    - Inserting vectors
    - Similarity search before update
    - UPDATE operation on indexed column
    - Similarity search after update (verify index updated)
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

        # Create index
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

        # Test similarity search before update
        await db_conn.execute("SET enable_seqscan = off")

        results_before = await db_conn.fetch("""
            SELECT * FROM vectors
            ORDER BY v1 <#> ARRAY[1.0, 2.0, 3.0]
            LIMIT 5
        """)

        # Verify expected order before update
        expected_ids_before = [1, 2, 3, 4]
        actual_ids_before = [row['id'] for row in results_before]
        assert actual_ids_before == expected_ids_before, \
            f"Expected order {expected_ids_before}, got {actual_ids_before}"

        # UPDATE vector for id=2
        await db_conn.execute("""
            UPDATE vectors SET v1 = ARRAY[9.0, 10.0, 11.0] WHERE id = 2
        """)

        # Test similarity search after update
        results_after = await db_conn.fetch("""
            SELECT * FROM vectors
            ORDER BY v1 <#> ARRAY[1.0, 2.0, 3.0]
            LIMIT 5
        """)

        # After update: id=2 has [9,10,11] which is farther from query [1,2,3] than id=3's [7,8,9]
        # L2 distances: id=1=0, id=3=10.4, id=2=13.9, id=4=441
        expected_ids_after = [1, 3, 2, 4]
        actual_ids_after = [row['id'] for row in results_after]
        assert actual_ids_after == expected_ids_after, \
            f"Expected order {expected_ids_after}, got {actual_ids_after}"

        # Verify v1 was updated but v2 wasn't
        row_2 = await db_conn.fetchrow("SELECT * FROM vectors WHERE id = 2")
        assert row_2['v1'] == [9.0, 10.0, 11.0], \
            f"v1 not updated correctly: {row_2['v1']}"
        assert row_2['v2'] == [4.0, 5.0, 6.0], \
            f"v2 should not change: {row_2['v2']}"

        print("✓ Test passed: UPDATE operations with index work correctly")

    finally:
        # Cleanup
        await db_conn.execute("RESET enable_seqscan")
        await db_conn.execute("DROP INDEX IF EXISTS index_for_v1 CASCADE")
        await db_conn.execute("DROP TABLE IF EXISTS vectors CASCADE")
