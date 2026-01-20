"""
Tests for constraint enforcement in pg_deeplake.

These tests verify that PRIMARY KEY, UNIQUE, and FOREIGN KEY constraints
work correctly with deeplake tables.
"""
import pytest
import asyncpg


@pytest.mark.asyncio
async def test_primary_key_rejects_duplicates(db_conn: asyncpg.Connection):
    """PRIMARY KEY should reject duplicate values."""
    try:
        await db_conn.execute("DROP TABLE IF EXISTS pk_test CASCADE")
        await db_conn.execute("""
            CREATE TABLE pk_test (id INT PRIMARY KEY, name TEXT) USING deeplake
        """)

        await db_conn.execute("INSERT INTO pk_test VALUES (1, 'alice')")

        with pytest.raises(asyncpg.UniqueViolationError):
            await db_conn.execute("INSERT INTO pk_test VALUES (1, 'bob')")

    finally:
        await db_conn.execute("DROP TABLE IF EXISTS pk_test CASCADE")


@pytest.mark.asyncio
async def test_unique_constraint_rejects_duplicates(db_conn: asyncpg.Connection):
    """UNIQUE constraint should reject duplicate values."""
    try:
        await db_conn.execute("DROP TABLE IF EXISTS unique_test CASCADE")
        await db_conn.execute("""
            CREATE TABLE unique_test (id INT PRIMARY KEY, email TEXT UNIQUE) USING deeplake
        """)

        await db_conn.execute("INSERT INTO unique_test VALUES (1, 'alice@test.com')")

        with pytest.raises(asyncpg.UniqueViolationError):
            await db_conn.execute("INSERT INTO unique_test VALUES (2, 'alice@test.com')")

    finally:
        await db_conn.execute("DROP TABLE IF EXISTS unique_test CASCADE")


@pytest.mark.asyncio
async def test_foreign_key_insert(db_conn: asyncpg.Connection):
    """INSERT into child table with FK should trigger parent lookup."""
    try:
        await db_conn.execute("DROP TABLE IF EXISTS fk_child CASCADE")
        await db_conn.execute("DROP TABLE IF EXISTS fk_parent CASCADE")

        await db_conn.execute("""
            CREATE TABLE fk_parent (id INT PRIMARY KEY) USING deeplake
        """)
        await db_conn.execute("""
            CREATE TABLE fk_child (
                id INT PRIMARY KEY,
                parent_id INT REFERENCES fk_parent(id)
            ) USING deeplake
        """)

        await db_conn.execute("INSERT INTO fk_parent VALUES (1)")
        await db_conn.execute("INSERT INTO fk_child VALUES (1, 1)")

    finally:
        await db_conn.execute("DROP TABLE IF EXISTS fk_child CASCADE")
        await db_conn.execute("DROP TABLE IF EXISTS fk_parent CASCADE")
