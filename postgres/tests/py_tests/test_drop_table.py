"""
Test DROP TABLE operations with deeplake storage and indexes.

Ported from: postgres/tests/sql/drop_table.sql
"""
import pytest
import asyncpg
from test_utils.assertions import Assertions


@pytest.mark.asyncio
async def test_drop_table_cleanup(db_conn: asyncpg.Connection):
    """
    Test that DROP TABLE properly cleans up indexes and datasets.

    Tests:
    - Creating table with deeplake_index
    - Verifying index exists in pg_class
    - Verifying index in pg_deeplake_metadata
    - Verifying dataset directory exists
    - DROP TABLE
    - Verifying index removed from pg_class
    - Verifying metadata cleaned up
    - Verifying dataset directory deleted
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
            CREATE INDEX index_for_v2 ON vectors USING deeplake_index (v2 DESC)
        """)

        # Verify index exists in pg_class
        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM pg_class WHERE relname = 'index_for_v2'"
        )

        # Verify index in metadata
        index_name = await db_conn.fetchval(
            "SELECT index_name FROM pg_deeplake_metadata"
        )
        assert index_name == 'index_for_v2', \
            f"Expected 'index_for_v2', got '{index_name}'"

        # Get dataset path and verify it exists
        dataset_path = await db_conn.fetchval(
            "SELECT ds_path FROM pg_deeplake_tables WHERE table_name = (SELECT table_name FROM pg_deeplake_metadata LIMIT 1)"
        )

        # Check if directory exists using pg_ls_dir
        dir_exists_before = await assertions.directory_exists(dataset_path)
        assert dir_exists_before, \
            f"Dataset directory '{dataset_path}' should exist before DROP TABLE"

        # DROP TABLE
        await db_conn.execute("DROP TABLE vectors CASCADE")

        # Verify index removed from pg_class
        await assertions.assert_query_row_count(
            0,
            "SELECT * FROM pg_class WHERE relname = 'index_for_v2'"
        )

        # Verify metadata cleaned up
        index_name_after = await db_conn.fetchval(
            "SELECT index_name FROM pg_deeplake_metadata"
        )
        assert index_name_after is None or index_name_after == '', \
            f"Metadata should be empty, got '{index_name_after}'"

        # Verify dataset directory deleted
        dir_exists_after = await assertions.directory_exists(dataset_path)
        assert not dir_exists_after, \
            f"Dataset directory '{dataset_path}' should not exist after DROP TABLE"

        print("✓ Test passed: DROP TABLE properly cleans up indexes and datasets")

    finally:
        # Cleanup (in case test fails)
        await db_conn.execute("DROP INDEX IF EXISTS index_for_v2 CASCADE")
        await db_conn.execute("DROP TABLE IF EXISTS vectors CASCADE")
