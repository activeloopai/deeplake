"""
Test reconnection handling and data persistence.

Ported from: postgres/tests/sql/reconnect_test.sql
"""
import pytest
import asyncpg
import os
from test_utils.assertions import Assertions


@pytest.mark.asyncio
@pytest.mark.reconnect
async def test_reconnect_preserves_data(pg_server):
    """
    Test that data and indexes persist after reconnection.

    This test:
    - Creates a table and index in first connection
    - Inserts test data
    - Closes the connection
    - Opens a new connection (simulating \c in psql)
    - Verifies data and index still exist
    - Verifies queries still work correctly
    """
    user = os.environ.get("USER", "postgres")

    # First connection
    conn1 = await asyncpg.connect(database="postgres", user=user, host="localhost")

    try:
        assertions1 = Assertions(conn1)

        # Setup extension
        await conn1.execute("DROP EXTENSION IF EXISTS pg_deeplake CASCADE")
        await conn1.execute("CREATE EXTENSION pg_deeplake")

        # Create table and index
        await conn1.execute("""
            CREATE TABLE vectors (
                id SERIAL PRIMARY KEY,
                v1 float4[],
                v2 float4[]
            ) USING deeplake
        """)
        await conn1.execute("""
            CREATE INDEX index_for_v1 ON vectors USING deeplake_index (v1 DESC)
        """)

        # Verify index exists
        await assertions1.assert_query_row_count(
            1,
            "SELECT * FROM pg_class WHERE relname = 'index_for_v1'"
        )

        # Insert test data
        await conn1.execute("""
            INSERT INTO vectors (v1, v2) VALUES
                (ARRAY[1.0, 2.0, 3.0], ARRAY[1.0, 2.0, 3.0]),
                (ARRAY[4.0, 5.0, 6.0], ARRAY[4.0, 5.0, 6.0]),
                (ARRAY[7.0, 8.0, 9.0], ARRAY[7.0, 8.0, 9.0]),
                (ARRAY[0.0, 0.0, 444], ARRAY[0.0, 0.0, 444])
        """)

        # Test index usage before disconnect
        await conn1.execute("SET enable_seqscan = off")

        # Create expected results
        await conn1.execute("""
            CREATE TEMP TABLE expected_vectors (id INTEGER, v1 REAL[], v2 REAL[])
        """)
        await conn1.execute("""
            INSERT INTO expected_vectors VALUES
                (1, '{1,2,3}', '{1,2,3}'),
                (2, '{4,5,6}', '{4,5,6}'),
                (3, '{7,8,9}', '{7,8,9}'),
                (4, '{0,0,444}', '{0,0,444}')
        """)

        # Verify query works
        results1 = await conn1.fetch("""
            SELECT * FROM vectors ORDER BY v1 <#> ARRAY[1.0, 2.0, 3.0] LIMIT 5
        """)
        assert len(results1) == 4, f"Expected 4 rows before reconnect, got {len(results1)}"

        # Get backend PID before disconnect
        pid1 = await conn1.fetchval("SELECT pg_backend_pid()")
        print(f"First connection PID: {pid1}")

    finally:
        # Close first connection (simulating disconnect)
        await conn1.close()

    # Second connection (simulating \c reconnect)
    conn2 = await asyncpg.connect(database="postgres", user=user, host="localhost")

    try:
        assertions2 = Assertions(conn2)

        # Get new backend PID
        pid2 = await conn2.fetchval("SELECT pg_backend_pid()")
        print(f"Second connection PID: {pid2}")

        # Verify we have a different backend
        assert pid1 != pid2, "PIDs should differ after reconnection"

        # Verify index still exists after reconnect
        await assertions2.assert_query_row_count(
            1,
            "SELECT * FROM pg_class WHERE relname = 'index_for_v1'"
        )

        # Test index usage after reconnect
        await conn2.execute("SET enable_seqscan = off")

        # Create expected results in new connection
        await conn2.execute("""
            CREATE TEMP TABLE expected_vectors (id INTEGER, v1 REAL[], v2 REAL[])
        """)
        await conn2.execute("""
            INSERT INTO expected_vectors VALUES
                (1, '{1,2,3}', '{1,2,3}'),
                (2, '{4,5,6}', '{4,5,6}'),
                (3, '{7,8,9}', '{7,8,9}'),
                (4, '{0,0,444}', '{0,0,444}')
        """)

        # Verify query still works after reconnect
        results2 = await conn2.fetch("""
            SELECT * FROM vectors ORDER BY v1 <#> ARRAY[1.0, 2.0, 3.0] LIMIT 5
        """)
        assert len(results2) == 4, f"Expected 4 rows after reconnect, got {len(results2)}"

        # Verify data matches
        for r1, r2 in zip(results1, results2):
            assert r1['id'] == r2['id'], (
                f"ID mismatch: {r1['id']} vs {r2['id']}"
            )
            assert r1['v1'] == r2['v1'], (
                f"v1 mismatch for id {r1['id']}: {r1['v1']} vs {r2['v1']}"
            )
            assert r1['v2'] == r2['v2'], (
                f"v2 mismatch for id {r1['id']}: {r1['v2']} vs {r2['v2']}"
            )

        print("✓ Test passed: Data and indexes persist after reconnection")

    finally:
        # Cleanup
        await conn2.execute("RESET enable_seqscan")
        await conn2.execute("DROP INDEX IF EXISTS index_for_v1 CASCADE")
        await conn2.execute("DROP TABLE IF EXISTS vectors CASCADE")
        await conn2.execute("DROP EXTENSION IF EXISTS pg_deeplake CASCADE")
        await conn2.close()
