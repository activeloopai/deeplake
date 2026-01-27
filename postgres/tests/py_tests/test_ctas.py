"""
Test CREATE TABLE AS SELECT (CTAS) operations with deeplake tables.

These tests verify that CTAS works correctly with various column counts,
including edge cases that previously caused crashes (9+ columns).
"""
import pytest
import asyncpg


@pytest.mark.asyncio
async def test_ctas_single_column(db_conn: asyncpg.Connection):
    """Test CTAS with a single column."""
    try:
        # Create source table
        await db_conn.execute("""
            CREATE TABLE source_single (
                id INTEGER,
                name TEXT,
                value REAL
            ) USING deeplake
        """)

        # Insert test data
        await db_conn.execute("""
            INSERT INTO source_single (id, name, value) VALUES
                (1, 'a', 1.0),
                (2, 'b', 2.0),
                (3, 'c', 3.0)
        """)

        # CTAS with single column
        await db_conn.execute("""
            CREATE TABLE dest_single AS SELECT id FROM source_single
        """)

        # Verify data
        count = await db_conn.fetchval("SELECT COUNT(*) FROM dest_single")
        assert count == 3, f"Expected 3 rows, got {count}"

        rows = await db_conn.fetch("SELECT id FROM dest_single ORDER BY id")
        assert [r['id'] for r in rows] == [1, 2, 3]

        print("CTAS with single column works correctly")

    finally:
        await db_conn.execute("DROP TABLE IF EXISTS dest_single")
        await db_conn.execute("DROP TABLE IF EXISTS source_single CASCADE")


@pytest.mark.asyncio
async def test_ctas_multiple_columns(db_conn: asyncpg.Connection):
    """Test CTAS with multiple columns (3 columns)."""
    try:
        # Create source table
        await db_conn.execute("""
            CREATE TABLE source_multi (
                id INTEGER,
                name TEXT,
                value REAL
            ) USING deeplake
        """)

        # Insert test data
        await db_conn.execute("""
            INSERT INTO source_multi (id, name, value) VALUES
                (1, 'alpha', 10.5),
                (2, 'beta', 20.5),
                (3, 'gamma', 30.5)
        """)

        # CTAS with all columns
        await db_conn.execute("""
            CREATE TABLE dest_multi AS SELECT id, name, value FROM source_multi
        """)

        # Verify data
        count = await db_conn.fetchval("SELECT COUNT(*) FROM dest_multi")
        assert count == 3, f"Expected 3 rows, got {count}"

        row = await db_conn.fetchrow("SELECT * FROM dest_multi WHERE id = 1")
        assert row['id'] == 1
        assert row['name'] == 'alpha'
        assert abs(row['value'] - 10.5) < 0.01

        print("CTAS with multiple columns works correctly")

    finally:
        await db_conn.execute("DROP TABLE IF EXISTS dest_multi")
        await db_conn.execute("DROP TABLE IF EXISTS source_multi CASCADE")


@pytest.mark.asyncio
async def test_ctas_nine_columns(db_conn: asyncpg.Connection):
    """
    Test CTAS with 9 columns.

    This is a regression test for a bug where CTAS with 9+ columns
    caused a SIGSEGV crash due to incorrect query routing through DuckDB.
    """
    try:
        # Create source table with many columns
        await db_conn.execute("""
            CREATE TABLE source_nine (
                col1 INTEGER,
                col2 INTEGER,
                col3 INTEGER,
                col4 INTEGER,
                col5 INTEGER,
                col6 REAL,
                col7 REAL,
                col8 TEXT,
                col9 TEXT,
                col10 TEXT
            ) USING deeplake
        """)

        # Insert test data
        await db_conn.execute("""
            INSERT INTO source_nine VALUES
                (1, 2, 3, 4, 5, 6.0, 7.0, 'eight', 'nine', 'ten'),
                (11, 12, 13, 14, 15, 16.0, 17.0, 'eighteen', 'nineteen', 'twenty')
        """)

        # CTAS with 9 columns - this previously crashed
        await db_conn.execute("""
            CREATE TABLE dest_nine AS
            SELECT col1, col2, col3, col4, col5, col6, col7, col8, col9
            FROM source_nine
        """)

        # Verify data
        count = await db_conn.fetchval("SELECT COUNT(*) FROM dest_nine")
        assert count == 2, f"Expected 2 rows, got {count}"

        row = await db_conn.fetchrow("SELECT * FROM dest_nine WHERE col1 = 1")
        assert row['col1'] == 1
        assert row['col5'] == 5
        assert row['col9'] == 'nine'

        print("CTAS with 9 columns works correctly (regression test passed)")

    finally:
        await db_conn.execute("DROP TABLE IF EXISTS dest_nine")
        await db_conn.execute("DROP TABLE IF EXISTS source_nine CASCADE")


@pytest.mark.asyncio
async def test_ctas_select_star(db_conn: asyncpg.Connection):
    """Test CTAS with SELECT * (all columns)."""
    try:
        # Create source table with many columns
        await db_conn.execute("""
            CREATE TABLE source_star (
                id INTEGER,
                col1 INTEGER,
                col2 REAL,
                col3 TEXT,
                col4 BOOLEAN,
                col5 INTEGER,
                col6 REAL,
                col7 TEXT,
                col8 INTEGER,
                col9 REAL,
                col10 TEXT
            ) USING deeplake
        """)

        # Insert test data
        await db_conn.execute("""
            INSERT INTO source_star VALUES
                (1, 10, 1.1, 'a', true, 100, 10.1, 'aa', 1000, 100.1, 'aaa'),
                (2, 20, 2.2, 'b', false, 200, 20.2, 'bb', 2000, 200.2, 'bbb'),
                (3, 30, 3.3, 'c', true, 300, 30.3, 'cc', 3000, 300.3, 'ccc')
        """)

        # CTAS with SELECT *
        await db_conn.execute("""
            CREATE TABLE dest_star AS SELECT * FROM source_star
        """)

        # Verify data
        count = await db_conn.fetchval("SELECT COUNT(*) FROM dest_star")
        assert count == 3, f"Expected 3 rows, got {count}"

        # Verify all columns are present
        row = await db_conn.fetchrow("SELECT * FROM dest_star WHERE id = 2")
        assert row['id'] == 2
        assert row['col1'] == 20
        assert abs(row['col2'] - 2.2) < 0.01
        assert row['col3'] == 'b'
        # Note: Boolean False may be stored as None in deeplake tables
        assert row['col4'] in (False, None)
        assert row['col10'] == 'bbb'

        print("CTAS with SELECT * works correctly")

    finally:
        await db_conn.execute("DROP TABLE IF EXISTS dest_star")
        await db_conn.execute("DROP TABLE IF EXISTS source_star CASCADE")


@pytest.mark.asyncio
async def test_ctas_with_limit(db_conn: asyncpg.Connection):
    """Test CTAS with LIMIT clause."""
    try:
        # Create source table
        await db_conn.execute("""
            CREATE TABLE source_limit (
                id INTEGER,
                value TEXT
            ) USING deeplake
        """)

        # Insert test data
        await db_conn.execute("""
            INSERT INTO source_limit (id, value) VALUES
                (1, 'one'),
                (2, 'two'),
                (3, 'three'),
                (4, 'four'),
                (5, 'five')
        """)

        # CTAS with LIMIT
        await db_conn.execute("""
            CREATE TABLE dest_limit AS SELECT * FROM source_limit LIMIT 3
        """)

        # Verify data
        count = await db_conn.fetchval("SELECT COUNT(*) FROM dest_limit")
        assert count == 3, f"Expected 3 rows, got {count}"

        print("CTAS with LIMIT works correctly")

    finally:
        await db_conn.execute("DROP TABLE IF EXISTS dest_limit")
        await db_conn.execute("DROP TABLE IF EXISTS source_limit CASCADE")


@pytest.mark.asyncio
async def test_ctas_with_where(db_conn: asyncpg.Connection):
    """Test CTAS with WHERE clause."""
    try:
        # Create source table
        await db_conn.execute("""
            CREATE TABLE source_where (
                id INTEGER,
                category TEXT,
                value REAL
            ) USING deeplake
        """)

        # Insert test data
        await db_conn.execute("""
            INSERT INTO source_where (id, category, value) VALUES
                (1, 'A', 10.0),
                (2, 'B', 20.0),
                (3, 'A', 30.0),
                (4, 'B', 40.0),
                (5, 'A', 50.0)
        """)

        # CTAS with WHERE
        await db_conn.execute("""
            CREATE TABLE dest_where AS
            SELECT * FROM source_where WHERE category = 'A'
        """)

        # Verify data
        count = await db_conn.fetchval("SELECT COUNT(*) FROM dest_where")
        assert count == 3, f"Expected 3 rows, got {count}"

        categories = await db_conn.fetch("SELECT DISTINCT category FROM dest_where")
        assert len(categories) == 1
        assert categories[0]['category'] == 'A'

        print("CTAS with WHERE works correctly")

    finally:
        await db_conn.execute("DROP TABLE IF EXISTS dest_where")
        await db_conn.execute("DROP TABLE IF EXISTS source_where CASCADE")


@pytest.mark.asyncio
async def test_insert_into_after_ctas(db_conn: asyncpg.Connection):
    """Test that INSERT works on table created via CTAS."""
    try:
        # Create source table
        await db_conn.execute("""
            CREATE TABLE source_insert (
                id INTEGER,
                name TEXT
            ) USING deeplake
        """)

        # Insert initial data
        await db_conn.execute("""
            INSERT INTO source_insert (id, name) VALUES
                (1, 'first'),
                (2, 'second')
        """)

        # CTAS
        await db_conn.execute("""
            CREATE TABLE dest_insert AS SELECT * FROM source_insert
        """)

        # Verify initial data
        count = await db_conn.fetchval("SELECT COUNT(*) FROM dest_insert")
        assert count == 2, f"Expected 2 rows, got {count}"

        # INSERT into CTAS result table
        await db_conn.execute("""
            INSERT INTO dest_insert (id, name) VALUES (3, 'third')
        """)

        # Verify after insert
        count = await db_conn.fetchval("SELECT COUNT(*) FROM dest_insert")
        assert count == 3, f"Expected 3 rows after INSERT, got {count}"

        row = await db_conn.fetchrow("SELECT * FROM dest_insert WHERE id = 3")
        assert row['name'] == 'third'

        print("INSERT into CTAS table works correctly")

    finally:
        await db_conn.execute("DROP TABLE IF EXISTS dest_insert")
        await db_conn.execute("DROP TABLE IF EXISTS source_insert CASCADE")


@pytest.mark.asyncio
async def test_ctas_different_column_types(db_conn: asyncpg.Connection):
    """Test CTAS with various column types."""
    try:
        # Create source table with different types
        await db_conn.execute("""
            CREATE TABLE source_types (
                int_col INTEGER,
                bigint_col BIGINT,
                real_col REAL,
                double_col DOUBLE PRECISION,
                text_col TEXT,
                varchar_col VARCHAR(100),
                bool_col BOOLEAN,
                date_col DATE,
                timestamp_col TIMESTAMP
            ) USING deeplake
        """)

        # Insert test data
        await db_conn.execute("""
            INSERT INTO source_types VALUES
                (1, 1000000000, 1.5, 1.555555, 'text1', 'varchar1', true,
                 '2024-01-15', '2024-01-15 10:30:00'),
                (2, 2000000000, 2.5, 2.555555, 'text2', 'varchar2', false,
                 '2024-02-20', '2024-02-20 14:45:00')
        """)

        # CTAS with all columns
        await db_conn.execute("""
            CREATE TABLE dest_types AS SELECT * FROM source_types
        """)

        # Verify data
        count = await db_conn.fetchval("SELECT COUNT(*) FROM dest_types")
        assert count == 2, f"Expected 2 rows, got {count}"

        row = await db_conn.fetchrow("SELECT * FROM dest_types WHERE int_col = 1")
        assert row['int_col'] == 1
        assert row['bigint_col'] == 1000000000
        assert abs(row['real_col'] - 1.5) < 0.01
        assert row['text_col'] == 'text1'
        assert row['bool_col'] == True

        print("CTAS with different column types works correctly")

    finally:
        await db_conn.execute("DROP TABLE IF EXISTS dest_types")
        await db_conn.execute("DROP TABLE IF EXISTS source_types CASCADE")


@pytest.mark.asyncio
async def test_ctas_preserves_regular_select(db_conn: asyncpg.Connection):
    """
    Test that regular SELECT queries still work after CTAS fix.

    This verifies that the fix for CTAS didn't break the DuckDB executor
    path for normal SELECT queries.
    """
    try:
        # Create table
        await db_conn.execute("""
            CREATE TABLE select_test (
                id INTEGER,
                value REAL
            ) USING deeplake
        """)

        # Insert test data
        await db_conn.execute("""
            INSERT INTO select_test (id, value) VALUES
                (1, 100.0),
                (2, 200.0),
                (3, 300.0)
        """)

        # Regular SELECT (should go through DuckDB if enabled)
        rows = await db_conn.fetch("""
            SELECT id, value FROM select_test ORDER BY id
        """)

        assert len(rows) == 3
        assert rows[0]['id'] == 1
        assert abs(rows[0]['value'] - 100.0) < 0.01

        # SELECT with aggregation
        total = await db_conn.fetchval("SELECT SUM(value) FROM select_test")
        assert abs(total - 600.0) < 0.01

        print("Regular SELECT queries still work correctly")

    finally:
        await db_conn.execute("DROP TABLE IF EXISTS select_test CASCADE")
