"""
Test for "cache lookup failed for type 0" error after DROP COLUMN.

This test validates the bug where dropping a column from a deeplake table
causes subsequent queries to fail with "cache lookup failed for type 0".

Root cause hypothesis:
When a column is dropped in PostgreSQL:
1. The column slot remains in TupleDesc but with attisdropped=true and atttypid=0
2. Deeplake's table_data caches base_typeids_ for ALL columns including dropped ones
3. When queries run, they try to lookup type info for OID 0, causing the error
"""
import pytest
import asyncpg


@pytest.mark.asyncio
async def test_drop_column_then_select(db_conn: asyncpg.Connection):
    """
    Test that SELECT works after dropping a column.

    Steps:
    1. Create a table with multiple columns
    2. Insert some data
    3. Drop one column
    4. SELECT from the table - should NOT fail with "cache lookup failed for type 0"
    """
    try:
        # Step 1: Create table with multiple columns
        await db_conn.execute("""
            CREATE TABLE test_drop_col (
                id SERIAL PRIMARY KEY,
                name TEXT,
                value INT,
                description TEXT
            ) USING deeplake
        """)

        # Step 2: Insert some data
        await db_conn.execute("""
            INSERT INTO test_drop_col (name, value, description)
            VALUES ('item1', 100, 'first item')
        """)
        await db_conn.execute("""
            INSERT INTO test_drop_col (name, value, description)
            VALUES ('item2', 200, 'second item')
        """)

        # Verify data is there
        count_before = await db_conn.fetchval("SELECT COUNT(*) FROM test_drop_col")
        assert count_before == 2, f"Expected 2 rows before drop, got {count_before}"

        # Step 3: Drop a column
        await db_conn.execute("ALTER TABLE test_drop_col DROP COLUMN description")

        # Check pg_attribute to verify the dropped column state
        attr_info = await db_conn.fetch("""
            SELECT attname, atttypid, attisdropped
            FROM pg_attribute
            WHERE attrelid = 'test_drop_col'::regclass
            AND attnum > 0
            ORDER BY attnum
        """)
        print("Column attributes after DROP:")
        for row in attr_info:
            print(f"  {row['attname']}: typid={row['atttypid']}, dropped={row['attisdropped']}")

        # Check deeplake metadata
        deeplake_info = await db_conn.fetch("""
            SELECT ds_path FROM pg_deeplake_tables
            WHERE table_name LIKE '%test_drop_col%'
        """)
        print(f"Deeplake tables: {deeplake_info}")

        # Try SELECT with explicit columns first (not *)
        print("Trying SELECT with explicit columns...")
        try:
            explicit_rows = await db_conn.fetch("SELECT id, name, value FROM test_drop_col")
            print(f"Explicit SELECT worked! Got {len(explicit_rows)} rows")
        except Exception as e:
            print(f"Explicit SELECT failed: {e}")

        # Step 4: SELECT from the table - this is where the bug manifests
        # Should NOT raise "cache lookup failed for type 0"
        rows = await db_conn.fetch("SELECT * FROM test_drop_col")
        assert len(rows) == 2, f"Expected 2 rows after drop, got {len(rows)}"

        # Verify the remaining columns work
        row = rows[0]
        assert 'name' in row.keys(), "name column should exist"
        assert 'value' in row.keys(), "value column should exist"
        assert 'description' not in row.keys(), "description column should not exist"

        print("SUCCESS: SELECT works after DROP COLUMN")

    finally:
        await db_conn.execute("DROP TABLE IF EXISTS test_drop_col CASCADE")


@pytest.mark.asyncio
async def test_drop_column_then_insert(db_conn: asyncpg.Connection):
    """
    Test that INSERT works after dropping a column.
    """
    try:
        # Create table
        await db_conn.execute("""
            CREATE TABLE test_drop_insert (
                id SERIAL PRIMARY KEY,
                name TEXT,
                extra TEXT
            ) USING deeplake
        """)

        # Insert initial data
        await db_conn.execute("INSERT INTO test_drop_insert (name, extra) VALUES ('before', 'drop')")

        # Drop column
        await db_conn.execute("ALTER TABLE test_drop_insert DROP COLUMN extra")

        # Insert after drop - should work
        await db_conn.execute("INSERT INTO test_drop_insert (name) VALUES ('after')")

        # Verify
        count = await db_conn.fetchval("SELECT COUNT(*) FROM test_drop_insert")
        assert count == 2, f"Expected 2 rows, got {count}"

        print("SUCCESS: INSERT works after DROP COLUMN")

    finally:
        await db_conn.execute("DROP TABLE IF EXISTS test_drop_insert CASCADE")


@pytest.mark.asyncio
async def test_drop_column_reconnect_then_select(db_conn: asyncpg.Connection, pg_server):
    """
    Test that SELECT works after dropping a column and reconnecting.

    This tests if the cached table_data is properly invalidated/reloaded
    when a new connection is made after a column drop.
    """
    import asyncpg
    import os

    try:
        # Create table and insert data
        await db_conn.execute("""
            CREATE TABLE test_drop_reconnect (
                id SERIAL PRIMARY KEY,
                col_a TEXT,
                col_b TEXT
            ) USING deeplake
        """)

        await db_conn.execute("INSERT INTO test_drop_reconnect (col_a, col_b) VALUES ('a1', 'b1')")

        # Drop column
        await db_conn.execute("ALTER TABLE test_drop_reconnect DROP COLUMN col_b")

        # Create a NEW connection (simulates what customer did)
        user = os.environ.get("USER", "postgres")
        new_conn = await asyncpg.connect(
            database="postgres",
            user=user,
            host="localhost",
            statement_cache_size=0
        )

        try:
            # Query from new connection - this is where stale cache would cause issues
            rows = await new_conn.fetch("SELECT * FROM test_drop_reconnect")
            assert len(rows) == 1, f"Expected 1 row, got {len(rows)}"
            assert 'col_a' in rows[0].keys()
            assert 'col_b' not in rows[0].keys()

            print("SUCCESS: SELECT works after DROP COLUMN + reconnect")
        finally:
            await new_conn.close()

    finally:
        await db_conn.execute("DROP TABLE IF EXISTS test_drop_reconnect CASCADE")


@pytest.mark.asyncio
async def test_drop_multiple_columns(db_conn: asyncpg.Connection):
    """
    Test that queries work after dropping multiple columns.
    """
    try:
        await db_conn.execute("""
            CREATE TABLE test_multi_drop (
                id SERIAL PRIMARY KEY,
                a TEXT,
                b TEXT,
                c TEXT,
                d TEXT
            ) USING deeplake
        """)

        await db_conn.execute("INSERT INTO test_multi_drop (a, b, c, d) VALUES ('1', '2', '3', '4')")

        # Drop multiple columns one by one
        await db_conn.execute("ALTER TABLE test_multi_drop DROP COLUMN b")

        # Query should work
        rows = await db_conn.fetch("SELECT * FROM test_multi_drop")
        assert len(rows) == 1

        await db_conn.execute("ALTER TABLE test_multi_drop DROP COLUMN d")

        # Query should still work
        rows = await db_conn.fetch("SELECT * FROM test_multi_drop")
        assert len(rows) == 1
        assert set(rows[0].keys()) == {'id', 'a', 'c'}

        print("SUCCESS: SELECT works after dropping multiple columns")

    finally:
        await db_conn.execute("DROP TABLE IF EXISTS test_multi_drop CASCADE")


@pytest.mark.asyncio
async def test_drop_column_with_array_type(db_conn: asyncpg.Connection):
    """
    Test dropping a column when table has array types.
    Array types use ARR_ELEMTYPE which could return 0 for dropped columns.
    """
    try:
        await db_conn.execute("""
            CREATE TABLE test_drop_array (
                id SERIAL PRIMARY KEY,
                embeddings FLOAT4[],
                metadata TEXT
            ) USING deeplake
        """)

        await db_conn.execute("""
            INSERT INTO test_drop_array (embeddings, metadata)
            VALUES (ARRAY[1.0, 2.0, 3.0]::FLOAT4[], 'test')
        """)

        # Drop non-array column
        await db_conn.execute("ALTER TABLE test_drop_array DROP COLUMN metadata")

        # Query should work
        rows = await db_conn.fetch("SELECT * FROM test_drop_array")
        assert len(rows) == 1
        assert 'embeddings' in rows[0].keys()

        print("SUCCESS: SELECT works after dropping column from table with arrays")

    finally:
        await db_conn.execute("DROP TABLE IF EXISTS test_drop_array CASCADE")
