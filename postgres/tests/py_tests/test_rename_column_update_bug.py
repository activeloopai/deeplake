"""
Test UPDATE operations after ALTER TABLE RENAME COLUMN.

This test isolates a known bug where UPDATE statements with WHERE clauses
on TEXT columns fail after renaming a column.
"""
import pytest
import asyncpg


@pytest.mark.asyncio
async def test_update_with_where_on_integer_after_rename(db_conn: asyncpg.Connection):
    """
    Test that UPDATE with WHERE on INTEGER columns works after RENAME COLUMN.
    This should pass.
    """
    try:
        await db_conn.execute("""
            CREATE TABLE test_update_int (
                id SERIAL PRIMARY KEY,
                name TEXT,
                email VARCHAR(255)
            ) USING deeplake
        """)

        await db_conn.execute("""
            INSERT INTO test_update_int (name, email) VALUES ('Alice', 'alice@example.com')
        """)

        await db_conn.execute("""
            ALTER TABLE test_update_int RENAME COLUMN email TO email_address
        """)

        # UPDATE with WHERE on INTEGER column - should work
        await db_conn.execute("""
            UPDATE test_update_int SET email_address = 'new@example.com' WHERE id = 1
        """)

        row = await db_conn.fetchrow("SELECT * FROM test_update_int WHERE id = 1")
        assert row is not None, "Row should exist"
        assert row['email_address'] == 'new@example.com', \
            f"Expected 'new@example.com', got '{row['email_address']}'"
        assert row['id'] == 1, f"Expected id=1, got {row['id']}"
        assert row['name'] == 'Alice', f"Expected 'Alice', got '{row['name']}'"

        print("✓ Test passed: UPDATE with WHERE on INTEGER after RENAME works")

    finally:
        try:
            await db_conn.execute("DROP TABLE IF EXISTS test_update_int CASCADE")
        except:
            pass


@pytest.mark.asyncio
async def test_update_with_where_on_text_after_rename(db_conn: asyncpg.Connection):
    """
    Test that UPDATE with WHERE on TEXT columns fails after RENAME COLUMN.

    Bug: UPDATE statements with WHERE clauses on TEXT columns after renaming
    a column cause the id column to become NULL, resulting in:
    ERROR: null value in column "id" violates not-null constraint
    DETAIL: Failing row contains (null, Alice, new@example.com).
    """
    try:
        await db_conn.execute("""
            CREATE TABLE test_update_text (
                id SERIAL PRIMARY KEY,
                name TEXT,
                email VARCHAR(255)
            ) USING deeplake
        """)

        await db_conn.execute("""
            INSERT INTO test_update_text (name, email) VALUES ('Alice', 'alice@example.com')
        """)

        await db_conn.execute("""
            ALTER TABLE test_update_text RENAME COLUMN email TO email_address
        """)

        # UPDATE with WHERE on TEXT column - currently fails
        await db_conn.execute("""
            UPDATE test_update_text SET email_address = 'new@example.com' WHERE name = 'Alice'
        """)

        row = await db_conn.fetchrow("SELECT * FROM test_update_text WHERE name = 'Alice'")
        assert row is not None, "Row should exist"
        assert row['email_address'] == 'new@example.com', \
            f"Expected 'new@example.com', got '{row['email_address']}'"
        assert row['id'] == 1, f"Expected id=1, got {row['id']}"
        assert row['name'] == 'Alice', f"Expected 'Alice', got '{row['name']}'"

        print("✓ Test passed: UPDATE with WHERE on TEXT after RENAME works")

    finally:
        try:
            await db_conn.execute("DROP TABLE IF EXISTS test_update_text CASCADE")
        except:
            pass


@pytest.mark.asyncio
async def test_update_without_where_after_rename(db_conn: asyncpg.Connection):
    """
    Test that UPDATE without WHERE clause works after RENAME COLUMN.
    This should pass.
    """
    try:
        await db_conn.execute("""
            CREATE TABLE test_update_no_where (
                id SERIAL PRIMARY KEY,
                name TEXT,
                email VARCHAR(255)
            ) USING deeplake
        """)

        await db_conn.execute("""
            INSERT INTO test_update_no_where (name, email) VALUES ('Alice', 'alice@example.com')
        """)

        await db_conn.execute("""
            ALTER TABLE test_update_no_where RENAME COLUMN email TO email_address
        """)

        # UPDATE without WHERE clause - should work
        await db_conn.execute("""
            UPDATE test_update_no_where SET email_address = 'new@example.com'
        """)

        row = await db_conn.fetchrow("SELECT * FROM test_update_no_where")
        assert row is not None, "Row should exist"
        assert row['email_address'] == 'new@example.com', \
            f"Expected 'new@example.com', got '{row['email_address']}'"
        assert row['id'] == 1, f"Expected id=1, got {row['id']}"
        assert row['name'] == 'Alice', f"Expected 'Alice', got '{row['name']}'"

        print("✓ Test passed: UPDATE without WHERE after RENAME works")

    finally:
        try:
            await db_conn.execute("DROP TABLE IF EXISTS test_update_no_where CASCADE")
        except:
            pass


@pytest.mark.asyncio
async def test_update_with_where_on_varchar_after_rename(db_conn: asyncpg.Connection):
    """
    Test that UPDATE with WHERE on VARCHAR columns also fails after RENAME COLUMN.
    This verifies the bug affects VARCHAR (not just TEXT).
    """
    try:
        await db_conn.execute("""
            CREATE TABLE test_update_varchar (
                id SERIAL PRIMARY KEY,
                name TEXT,
                email VARCHAR(255)
            ) USING deeplake
        """)

        await db_conn.execute("""
            INSERT INTO test_update_varchar (name, email) VALUES ('Alice', 'alice@example.com')
        """)

        await db_conn.execute("""
            ALTER TABLE test_update_varchar RENAME COLUMN name TO full_name
        """)

        # UPDATE with WHERE on VARCHAR column - currently fails
        await db_conn.execute("""
            UPDATE test_update_varchar SET email = 'new@example.com' WHERE email = 'alice@example.com'
        """)

        row = await db_conn.fetchrow("SELECT * FROM test_update_varchar WHERE email = 'new@example.com'")
        assert row is not None, "Row should exist"
        assert row['email'] == 'new@example.com', \
            f"Expected 'new@example.com', got '{row['email']}'"
        assert row['id'] == 1, f"Expected id=1, got {row['id']}"
        assert row['full_name'] == 'Alice', f"Expected 'Alice', got '{row['full_name']}'"

        print("✓ Test passed: UPDATE with WHERE on VARCHAR after RENAME works")

    finally:
        try:
            await db_conn.execute("DROP TABLE IF EXISTS test_update_varchar CASCADE")
        except:
            pass
