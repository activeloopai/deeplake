"""
Test DROP TABLE when the underlying deeplake dataset has been deleted externally.

This test verifies that dropping a table succeeds silently when the underlying
dataset has already been deleted via the deeplake API.
"""
import pytest
import asyncpg
import deeplake
import os


@pytest.mark.asyncio
async def test_drop_table_with_missing_dataset(db_conn: asyncpg.Connection, temp_dir_for_postgres):
    """
    Test that DROP TABLE succeeds silently when the underlying dataset is already deleted.

    Steps:
    1. Create a table with a specific dataset_path
    2. Delete the dataset directly using deeplake API
    3. DROP TABLE should succeed without error
    """
    dataset_path = os.path.join(temp_dir_for_postgres, "test_dataset")

    # Create table with explicit dataset path
    await db_conn.execute(f"""
        CREATE TABLE test_missing_ds (
            id SERIAL PRIMARY KEY,
            name TEXT,
            value INT
        ) USING deeplake WITH (dataset_path = '{dataset_path}')
    """)
    print(f"✓ Created table with dataset_path: {dataset_path}")

    # Verify the dataset was created
    assert os.path.exists(dataset_path), \
        f"Dataset should exist at {dataset_path}"
    print("✓ Dataset directory exists")

    # Insert some data to make it a real dataset
    await db_conn.execute("""
        INSERT INTO test_missing_ds (name, value) VALUES ('test', 42)
    """)
    print("✓ Inserted test data")

    # Delete the dataset directly using deeplake API
    deeplake.delete(dataset_path)
    print("✓ Deleted dataset via deeplake API")

    # Verify the dataset is gone
    assert not os.path.exists(dataset_path), \
        f"Dataset should no longer exist at {dataset_path}"
    print("✓ Verified dataset is deleted")

    # Now DROP TABLE should succeed silently without error
    await db_conn.execute("DROP TABLE test_missing_ds CASCADE")
    print("✓ DROP TABLE succeeded (dataset was already deleted)")

    # Verify the table is gone from metadata
    table_exists = await db_conn.fetchval("""
        SELECT COUNT(*) FROM pg_deeplake_tables
        WHERE table_name = 'public.test_missing_ds'
    """)
    assert table_exists == 0, "Table should be removed from metadata"
    print("✓ Table metadata cleaned up")


@pytest.mark.asyncio
async def test_drop_table_normal_vs_missing_dataset(db_conn: asyncpg.Connection, temp_dir_for_postgres):
    """
    Test that DROP TABLE behaves the same whether dataset exists or not.

    Creates two tables:
    1. One where we drop normally (dataset exists)
    2. One where we delete the dataset first, then drop

    Both should succeed without error.
    """
    dataset_path_normal = os.path.join(temp_dir_for_postgres, "normal_dataset")
    dataset_path_deleted = os.path.join(temp_dir_for_postgres, "deleted_dataset")

    # Create both tables
    await db_conn.execute(f"""
        CREATE TABLE test_normal (
            id INT PRIMARY KEY
        ) USING deeplake WITH (dataset_path = '{dataset_path_normal}')
    """)
    await db_conn.execute(f"""
        CREATE TABLE test_deleted (
            id INT PRIMARY KEY
        ) USING deeplake WITH (dataset_path = '{dataset_path_deleted}')
    """)
    print("✓ Created both tables")

    # Verify both datasets exist
    assert os.path.exists(dataset_path_normal), "Normal dataset should exist"
    assert os.path.exists(dataset_path_deleted), "Deleted dataset should exist"
    print("✓ Both datasets exist")

    # Delete one dataset externally
    deeplake.delete(dataset_path_deleted)
    assert not os.path.exists(dataset_path_deleted), "Deleted dataset should be gone"
    print("✓ Deleted one dataset via deeplake API")

    # Drop the table with missing dataset (should succeed silently)
    await db_conn.execute("DROP TABLE test_deleted CASCADE")
    print("✓ DROP TABLE with missing dataset succeeded")

    # Drop the table with existing dataset (normal case)
    await db_conn.execute("DROP TABLE test_normal CASCADE")
    print("✓ DROP TABLE with existing dataset succeeded")

    # Verify both datasets are cleaned up
    assert not os.path.exists(dataset_path_normal), \
        "Normal dataset should be deleted by DROP TABLE"
    print("✓ Normal dataset was cleaned up by DROP TABLE")

    # Verify both tables are gone from metadata
    table_count = await db_conn.fetchval("""
        SELECT COUNT(*) FROM pg_deeplake_tables
        WHERE table_name IN ('public.test_normal', 'public.test_deleted')
    """)
    assert table_count == 0, "Both tables should be removed from metadata"
    print("✓ Both table metadata entries cleaned up")

