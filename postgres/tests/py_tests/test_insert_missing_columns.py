"""
Test INSERT operations with missing columns.

This test suite verifies that the extension correctly handles INSERT statements
where fewer values are provided than columns exist in the table. PostgreSQL
automatically fills missing columns with NULL (or defaults), and the extension
must handle these NULL values correctly.

Regression test for bug where inserting with missing columns caused crash:
  "Data of type Text must have '0' dimensions provided '1'"

The issue was in table_data_impl.hpp where NULL values for text columns were
created with dimension 1 instead of 0.
"""
import pytest
import asyncpg


@pytest.mark.asyncio
async def test_insert_missing_trailing_columns(db_conn: asyncpg.Connection):
    """
    Test INSERT with missing trailing columns.

    Verifies that when fewer values are provided than columns exist,
    PostgreSQL fills in NULLs and the extension handles them correctly.
    """
    try:
        # Create table with 3 columns
        await db_conn.execute("""
            CREATE TABLE users (
                id INT,
                name TEXT,
                email TEXT
            ) USING deeplake
        """)

        # Insert with only 2 values (email should be NULL)
        await db_conn.execute("INSERT INTO users VALUES (1, 'sasun')")

        # Verify the row was inserted
        result = await db_conn.fetch("SELECT * FROM users")
        assert len(result) == 1, f"Expected 1 row, got {len(result)}"

        row = result[0]
        assert row['id'] == 1, f"Expected id=1, got {row['id']}"
        assert row['name'] == 'sasun', f"Expected name='sasun', got {row['name']}"
        # Text NULL values are stored/returned as empty strings in deeplake
        assert row['email'] == '', f"Expected email='', got {row['email']}"

        print("✓ Test passed: Insert with missing trailing column")

    finally:
        await db_conn.execute("DROP TABLE IF EXISTS users CASCADE")


@pytest.mark.asyncio
async def test_insert_missing_multiple_columns(db_conn: asyncpg.Connection):
    """
    Test INSERT with multiple missing columns.

    Verifies that multiple trailing NULL columns are handled correctly.
    """
    try:
        # Create table with 5 columns
        await db_conn.execute("""
            CREATE TABLE products (
                id INT,
                name TEXT,
                description TEXT,
                category TEXT,
                price FLOAT
            ) USING deeplake
        """)

        # Insert with only 2 values (description, category, price should be NULL)
        await db_conn.execute("INSERT INTO products VALUES (1, 'Widget')")

        # Verify the row was inserted
        result = await db_conn.fetch("SELECT * FROM products")
        assert len(result) == 1, f"Expected 1 row, got {len(result)}"

        row = result[0]
        assert row['id'] == 1
        assert row['name'] == 'Widget'
        # Text NULL values are stored/returned as empty strings
        assert row['description'] == ''
        assert row['category'] == ''
        assert row['price'] == 0  # Numeric NULLs are stored as 0 in deeplake

        print("✓ Test passed: Insert with multiple missing columns")

    finally:
        await db_conn.execute("DROP TABLE IF EXISTS products CASCADE")


@pytest.mark.asyncio
async def test_insert_all_columns_then_missing(db_conn: asyncpg.Connection):
    """
    Test mixed INSERT operations with and without missing columns.

    Verifies that we can insert complete rows and incomplete rows
    into the same table.
    """
    try:
        # Create table
        await db_conn.execute("""
            CREATE TABLE contacts (
                id INT,
                name TEXT,
                phone TEXT,
                email TEXT
            ) USING deeplake
        """)

        # Insert complete row
        await db_conn.execute(
            "INSERT INTO contacts VALUES (1, 'Alice', '555-1234', 'alice@example.com')"
        )

        # Insert row with missing columns
        await db_conn.execute("INSERT INTO contacts VALUES (2, 'Bob')")

        # Insert another complete row
        await db_conn.execute(
            "INSERT INTO contacts VALUES (3, 'Charlie', '555-5678', 'charlie@example.com')"
        )

        # Insert row with only 3 columns
        await db_conn.execute("INSERT INTO contacts VALUES (4, 'David', '555-9999')")

        # Verify all rows
        result = await db_conn.fetch("SELECT * FROM contacts ORDER BY id")
        assert len(result) == 4, f"Expected 4 rows, got {len(result)}"

        # Check first row (complete)
        assert result[0]['id'] == 1
        assert result[0]['name'] == 'Alice'
        assert result[0]['phone'] == '555-1234'
        assert result[0]['email'] == 'alice@example.com'

        # Check second row (missing phone and email)
        assert result[1]['id'] == 2
        assert result[1]['name'] == 'Bob'
        assert result[1]['phone'] == ''  # Text NULL
        assert result[1]['email'] == ''  # Text NULL

        # Check third row (complete)
        assert result[2]['id'] == 3
        assert result[2]['name'] == 'Charlie'
        assert result[2]['phone'] == '555-5678'
        assert result[2]['email'] == 'charlie@example.com'

        # Check fourth row (missing email)
        assert result[3]['id'] == 4
        assert result[3]['name'] == 'David'
        assert result[3]['phone'] == '555-9999'
        assert result[3]['email'] == ''  # Text NULL

        print("✓ Test passed: Mixed complete and incomplete inserts")

    finally:
        await db_conn.execute("DROP TABLE IF EXISTS contacts CASCADE")


@pytest.mark.asyncio
async def test_insert_numeric_and_text_nulls(db_conn: asyncpg.Connection):
    """
    Test NULL handling for both numeric and text columns.

    Verifies that NULL values are correctly handled for different data types:
    - Numeric columns: stored as 0
    - Text columns: stored as NULL
    """
    try:
        # Create table with mixed types
        await db_conn.execute("""
            CREATE TABLE mixed_types (
                int_col INT,
                text_col TEXT,
                float_col FLOAT,
                text_col2 TEXT
            ) USING deeplake
        """)

        # Insert with only first column
        await db_conn.execute("INSERT INTO mixed_types VALUES (42)")

        # Verify
        result = await db_conn.fetch("SELECT * FROM mixed_types")
        assert len(result) == 1

        row = result[0]
        assert row['int_col'] == 42
        assert row['text_col'] == ''  # Text NULL stored as empty string
        assert row['float_col'] == 0  # Numeric NULL stored as 0
        assert row['text_col2'] == ''  # Text NULL stored as empty string

        print("✓ Test passed: Numeric and text NULL handling")

    finally:
        await db_conn.execute("DROP TABLE IF EXISTS mixed_types CASCADE")


@pytest.mark.asyncio
async def test_insert_single_column_value(db_conn: asyncpg.Connection):
    """
    Test INSERT with only one value for multi-column table.

    This is the original bug scenario from the issue report.
    """
    try:
        # Create table with 3 columns (int, text, text)
        await db_conn.execute("""
            CREATE TABLE users (
                id INT,
                name TEXT,
                email TEXT
            ) USING deeplake
        """)

        # This was causing crash: INSERT with only 1 value for 3-column table
        await db_conn.execute("INSERT INTO users VALUES (1)")

        # Verify
        result = await db_conn.fetch("SELECT * FROM users")
        assert len(result) == 1

        row = result[0]
        assert row['id'] == 1
        assert row['name'] == ''  # Text NULL stored as empty string
        assert row['email'] == ''  # Text NULL stored as empty string

        print("✓ Test passed: Single value insert for multi-column table")

    finally:
        await db_conn.execute("DROP TABLE IF EXISTS users CASCADE")
