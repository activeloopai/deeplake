"""
Test deeplake.root_path session variable functionality.

Tests session-level root path configuration for automatic table path resolution.
"""
import pytest
import asyncpg
import tempfile
import os
from pathlib import Path


@pytest.mark.asyncio
async def test_root_path_basic(db_conn: asyncpg.Connection, temp_dir_for_postgres):
    """
    Test basic root_path functionality.

    Tests:
    - Setting deeplake.root_path
    - Showing deeplake.root_path
    - Creating table without explicit path uses root_path
    - Table is created at root_path/schema/table location
    """
    # Use the temp directory with proper permissions
    root_path = temp_dir_for_postgres

    # Set root path
    await db_conn.execute(f"SET deeplake.root_path = '{root_path}'")

    # Verify root path is set
    current_root = await db_conn.fetchval("SHOW deeplake.root_path")
    assert current_root == root_path, \
        f"Expected root_path '{root_path}', got '{current_root}'"
    print(f"✓ Set deeplake.root_path to: {root_path}")

    try:
        # Create table without explicit path
        await db_conn.execute("""
            CREATE TABLE test_root_path (
                id INT,
                name TEXT
            ) USING deeplake
        """)
        print("✓ Created table without explicit path")

        # Verify table was created
        row_count = await db_conn.fetchval("SELECT COUNT(*) FROM test_root_path")
        assert row_count == 0, "Table should be empty after creation"

        # Get the dataset path from metadata
        ds_path = await db_conn.fetchval("""
            SELECT ds_path FROM pg_deeplake_tables
            WHERE table_name = 'public.test_root_path'
        """)
        assert ds_path is not None, "Dataset path should exist in metadata"
        print(f"✓ Dataset path from metadata: {ds_path}")

        # Verify the path starts with our root_path
        expected_path = os.path.join(root_path, "public", "test_root_path")
        # Strip trailing slash for comparison
        ds_path_normalized = ds_path.rstrip('/')
        expected_path_normalized = expected_path.rstrip('/')
        assert ds_path_normalized == expected_path_normalized, \
            f"Expected path '{expected_path}', got '{ds_path}'"
        print(f"✓ Table created at expected location: {ds_path}")

        # Verify the dataset directory exists
        assert os.path.exists(ds_path), \
            f"Dataset directory should exist at {ds_path}"
        print("✓ Dataset directory exists")

        # Test basic operations
        await db_conn.execute("""
            INSERT INTO test_root_path VALUES (1, 'test')
        """)
        count = await db_conn.fetchval("SELECT COUNT(*) FROM test_root_path")
        assert count == 1, "Should have 1 row after insert"
        print("✓ Basic insert/select operations work")

    finally:
        # Cleanup
        await db_conn.execute("DROP TABLE IF EXISTS test_root_path CASCADE")
        await db_conn.execute("RESET deeplake.root_path")


@pytest.mark.asyncio
async def test_root_path_explicit_override(db_conn: asyncpg.Connection, temp_dir_for_postgres):
    """
    Test that explicit WITH clause overrides root_path.

    Tests:
    - Setting root_path
    - Creating table with explicit path
    - Table ignores root_path and uses explicit path
    """
    tmpdir = temp_dir_for_postgres
    root_path = os.path.join(tmpdir, "root")
    explicit_path = os.path.join(tmpdir, "explicit")
    os.makedirs(root_path, exist_ok=True)
    os.makedirs(explicit_path, exist_ok=True)

    # Set root path
    await db_conn.execute(f"SET deeplake.root_path = '{root_path}'")
    print(f"✓ Set deeplake.root_path to: {root_path}")

    try:
        # Create table with explicit path (should override root_path)
        await db_conn.execute(f"""
            CREATE TABLE test_explicit (
                id INT,
                value TEXT
            ) USING deeplake WITH (dataset_path = '{explicit_path}')
        """)
        print(f"✓ Created table with explicit path: {explicit_path}")

        # Get the dataset path from metadata
        ds_path = await db_conn.fetchval("""
            SELECT ds_path FROM pg_deeplake_tables
            WHERE table_name = 'public.test_explicit'
        """)

        # Verify it used the explicit path, not root_path
        # Strip trailing slash for comparison
        ds_path_normalized = ds_path.rstrip('/')
        explicit_path_normalized = explicit_path.rstrip('/')
        assert ds_path_normalized == explicit_path_normalized, \
            f"Expected explicit path '{explicit_path}', got '{ds_path}'"
        print("✓ Table correctly used explicit path (not root_path)")

        # Verify dataset exists at explicit path
        assert os.path.exists(explicit_path), \
            f"Dataset should exist at explicit path {explicit_path}"
        print("✓ Dataset created at explicit location")

    finally:
        # Cleanup
        await db_conn.execute("DROP TABLE IF EXISTS test_explicit CASCADE")
        await db_conn.execute("RESET deeplake.root_path")


@pytest.mark.asyncio
async def test_root_path_reset(db_conn: asyncpg.Connection):
    """
    Test resetting root_path.

    Tests:
    - Setting root_path
    - Resetting root_path
    - After reset, tables use default location
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        root_path = tmpdir

        # Set root path
        await db_conn.execute(f"SET deeplake.root_path = '{root_path}'")
        current = await db_conn.fetchval("SHOW deeplake.root_path")
        assert current == root_path, "Root path should be set"
        print(f"✓ Set root_path to: {root_path}")

        # Reset root path
        await db_conn.execute("RESET deeplake.root_path")
        current = await db_conn.fetchval("SHOW deeplake.root_path")
        assert current == "", "Root path should be empty after reset"
        print("✓ Reset root_path (now empty)")

        try:
            # Create table after reset - should use default location (pg data dir)
            await db_conn.execute("""
                CREATE TABLE test_after_reset (
                    id INT
                ) USING deeplake
            """)

            # Get the dataset path
            ds_path = await db_conn.fetchval("""
                SELECT ds_path FROM pg_deeplake_tables
                WHERE table_name = 'public.test_after_reset'
            """)

            # Should NOT be in our temp root_path
            assert not ds_path.startswith(root_path), \
                f"Path should not start with reset root_path: {ds_path}"
            print(f"✓ After reset, table uses default location: {ds_path}")

        finally:
            # Cleanup
            await db_conn.execute("DROP TABLE IF EXISTS test_after_reset CASCADE")


@pytest.mark.asyncio
async def test_root_path_empty_string(db_conn: asyncpg.Connection):
    """
    Test setting root_path to empty string.

    Tests:
    - Setting root_path to empty string
    - Tables use default location
    """
    # Set root path to empty string
    await db_conn.execute("SET deeplake.root_path = ''")
    current = await db_conn.fetchval("SHOW deeplake.root_path")
    assert current == "", "Root path should be empty"
    print("✓ Set root_path to empty string")

    try:
        # Create table with empty root_path
        await db_conn.execute("""
            CREATE TABLE test_empty_root (
                id INT
            ) USING deeplake
        """)

        # Get the dataset path
        ds_path = await db_conn.fetchval("""
            SELECT ds_path FROM pg_deeplake_tables
            WHERE table_name = 'public.test_empty_root'
        """)

        # Should use default location (not empty)
        assert ds_path != "", "Dataset path should not be empty"
        assert os.path.exists(ds_path), "Dataset should exist at default location"
        print(f"✓ With empty root_path, uses default location: {ds_path}")

    finally:
        # Cleanup
        await db_conn.execute("DROP TABLE IF EXISTS test_empty_root CASCADE")
        await db_conn.execute("RESET deeplake.root_path")


@pytest.mark.asyncio
async def test_root_path_multiple_tables(db_conn: asyncpg.Connection, temp_dir_for_postgres):
    """
    Test creating multiple tables with same root_path.

    Tests:
    - Setting root_path once
    - Creating multiple tables
    - All tables use the same root_path
    """
    root_path = temp_dir_for_postgres

    # Set root path
    await db_conn.execute(f"SET deeplake.root_path = '{root_path}'")
    print(f"✓ Set root_path to: {root_path}")

    try:
        # Create multiple tables
        await db_conn.execute("""
            CREATE TABLE test_multi1 (id INT) USING deeplake
        """)
        await db_conn.execute("""
            CREATE TABLE test_multi2 (id INT) USING deeplake
        """)
        await db_conn.execute("""
            CREATE TABLE test_multi3 (id INT) USING deeplake
        """)
        print("✓ Created 3 tables")

        # Get all dataset paths
        paths = await db_conn.fetch("""
            SELECT table_name, ds_path FROM pg_deeplake_tables
            WHERE table_name IN ('public.test_multi1', 'public.test_multi2', 'public.test_multi3')
            ORDER BY table_name
        """)

        assert len(paths) == 3, "Should have 3 tables"

        # Verify all use the root_path
        for row in paths:
            table_name = row['table_name']
            ds_path = row['ds_path']
            assert ds_path.startswith(root_path), \
                f"Table {table_name} path should start with root_path: {ds_path}"
            print(f"✓ {table_name}: {ds_path}")

        # Verify all directories exist
        for row in paths:
            assert os.path.exists(row['ds_path']), \
                f"Directory should exist: {row['ds_path']}"
        print("✓ All dataset directories exist")

    finally:
        # Cleanup
        await db_conn.execute("DROP TABLE IF EXISTS test_multi1 CASCADE")
        await db_conn.execute("DROP TABLE IF EXISTS test_multi2 CASCADE")
        await db_conn.execute("DROP TABLE IF EXISTS test_multi3 CASCADE")
        await db_conn.execute("RESET deeplake.root_path")
