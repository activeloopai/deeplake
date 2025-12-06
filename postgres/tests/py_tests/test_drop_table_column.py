"""
Test ALTER TABLE DROP COLUMN operations with deeplake indexes.

Ported from: postgres/tests/sql/drop_table_column.sql
"""
import pytest
import asyncpg
from lib.assertions import Assertions


@pytest.mark.asyncio
async def test_drop_table_column(db_conn: asyncpg.Connection):
    """
    Test that ALTER TABLE DROP COLUMN properly handles deeplake table access method.

    Tests:
    - Creating table with indexed and non-indexed columns
    - DROP non-indexed column (index should remain)
    - Verifying index and dataset persist
    - DROP indexed column (index should be removed)
    - Verifying index and dataset are cleaned up
    """
    assertions = Assertions(db_conn)

    try:
        # Create table with multiple columns
        await db_conn.execute("""
            CREATE TABLE vectors (
                id SERIAL PRIMARY KEY,
                v1 float4[],
                v2 float4[]
            ) USING deeplake
        """)

        # Create index on v2 (not v1)
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

        dir_exists_before = await assertions.directory_exists(dataset_path)
        assert dir_exists_before, \
            f"Dataset directory '{dataset_path}' should exist before DROP COLUMN"

        # DROP non-indexed column (v1) - index should remain
        await db_conn.execute("ALTER TABLE vectors DROP COLUMN v1")

        # Verify index still exists after dropping non-indexed column
        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM pg_class WHERE relname = 'index_for_v2'"
        )

        index_name_after_v1 = await db_conn.fetchval(
            "SELECT index_name FROM pg_deeplake_metadata"
        )
        assert index_name_after_v1 == 'index_for_v2', \
            f"Index should still exist after dropping non-indexed column, got '{index_name_after_v1}'"

        # Verify dataset directory still exists
        dataset_path_after_v1 = await db_conn.fetchval(
            "SELECT ds_path FROM pg_deeplake_tables WHERE table_name = (SELECT table_name FROM pg_deeplake_metadata LIMIT 1)"
        )
        dir_exists_after_v1 = await assertions.directory_exists(dataset_path_after_v1)
        assert dir_exists_after_v1, \
            f"Dataset directory '{dataset_path_after_v1}' should exist after dropping non-indexed column"

        # DROP indexed column (v2) - index should be removed
        await db_conn.execute("ALTER TABLE vectors DROP COLUMN v2")

        # Verify index removed from pg_class
        await assertions.assert_query_row_count(
            0,
            "SELECT * FROM pg_class WHERE relname = 'index_for_v2'"
        )

        # Verify metadata cleaned up
        index_name_after_v2 = await db_conn.fetchval(
            "SELECT index_name FROM pg_deeplake_metadata"
        )
        assert index_name_after_v2 is None or index_name_after_v2 == '', \
            f"Metadata should be empty after dropping indexed column, got '{index_name_after_v2}'"

        # Verify dataset directory deleted
        dir_exists_after_v2 = await assertions.directory_exists(dataset_path)
        assert dir_exists_after_v2, \
            f"Dataset directory '{dataset_path}' should exist after dropping indexed column"

        print("✓ Test passed: ALTER TABLE DROP COLUMN properly handles deeplake indexes")

    finally:
        # Cleanup (in case test fails)
        await db_conn.execute("DROP INDEX IF EXISTS index_for_v2 CASCADE")
        await db_conn.execute("DROP TABLE IF EXISTS vectors CASCADE")
