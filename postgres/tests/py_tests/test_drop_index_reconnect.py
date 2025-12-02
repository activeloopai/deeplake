"""
Test DROP INDEX operations after reconnection.

Ported from: postgres/tests/sql/drop_index_reconnect.sql
"""
import pytest
import asyncpg
import os
from lib.assertions import Assertions


@pytest.mark.asyncio
@pytest.mark.reconnect
async def test_drop_index_after_reconnect(pg_server):
    """
    Test that DROP INDEX works correctly after reconnection.

    Tests:
    - Creating table and index with custom dataset_path
    - Verifying index and dataset exist
    - Disconnecting and reconnecting
    - DROP INDEX after reconnection
    - Verifying index and dataset are cleaned up
    """
    user = os.environ.get("USER", "postgres")
    ds_path = "current_dataset/"

    # First connection
    conn1 = await asyncpg.connect(database="postgres", user=user, host="localhost")

    try:
        assertions1 = Assertions(conn1)

        # Setup extension
        await conn1.execute("DROP EXTENSION IF EXISTS pg_deeplake CASCADE")
        await conn1.execute("CREATE EXTENSION pg_deeplake")

        # Create table
        await conn1.execute("""
            CREATE TABLE vectors (
                id SERIAL PRIMARY KEY,
                v1 float4[],
                v2 float4[]
            ) USING deeplake WITH (dataset_path = 'current_dataset')
        """)

        # Create index with custom dataset_path
        create_index_query = f"""
            CREATE INDEX index_for_v1 ON vectors
            USING deeplake_index (v1 DESC)
        """
        await conn1.execute(create_index_query)

        # Verify index exists
        await assertions1.assert_query_row_count(
            1,
            "SELECT * FROM pg_class WHERE relname = 'index_for_v1'"
        )

        # Verify metadata
        index_name = await conn1.fetchval(
            "SELECT index_name FROM pg_deeplake_metadata"
        )
        assert index_name == 'index_for_v1', \
            f"Expected 'index_for_v1', got '{index_name}'"

        # Verify dataset directory exists and is not empty
        dir_exists = await assertions1.directory_exists(ds_path)
        assert dir_exists, f"Dataset directory '{ds_path}' should exist"

        dir_empty = await assertions1.directory_empty(ds_path)
        assert not dir_empty, f"Dataset directory '{ds_path}' should not be empty"

    finally:
        await conn1.close()

    # Second connection (after reconnect)
    conn2 = await asyncpg.connect(database="postgres", user=user, host="localhost")

    try:
        assertions2 = Assertions(conn2)

        # DROP INDEX after reconnect
        await conn2.execute("DROP INDEX index_for_v1 CASCADE")

        # Verify index removed from pg_class
        await assertions2.assert_query_row_count(
            0,
            "SELECT * FROM pg_class WHERE relname = 'index_for_v1'"
        )

        # Verify metadata cleaned up
        index_name_after = await conn2.fetchval(
            "SELECT index_name FROM pg_deeplake_metadata"
        )
        assert index_name_after is None or index_name_after == '', \
            f"Metadata should be empty, got '{index_name_after}'"

        # Verify dataset directory is cleaned up (empty)
        dir_empty_after = await assertions2.directory_empty(ds_path)
        assert not dir_empty_after, \
            f"Dataset directory '{ds_path}' should not be empty after DROP INDEX"

        print("✓ Test passed: DROP INDEX after reconnect works correctly")

    finally:
        # Cleanup
        await conn2.execute("DROP INDEX IF EXISTS index_for_v1 CASCADE")
        await conn2.execute("DROP TABLE IF EXISTS vectors CASCADE")
        await conn2.execute("DROP EXTENSION IF EXISTS pg_deeplake CASCADE")
        await conn2.close()
