"""
Test FILE domain type (domain over BYTEA, maps to link-of-bytes in deeplake).
"""
import pytest
import asyncpg
from test_utils.assertions import Assertions


@pytest.mark.asyncio
async def test_file_domain(db_conn: asyncpg.Connection):
    """
    Test FILE domain type for link-of-bytes data.

    Tests:
    - Creating FILE columns (domain over BYTEA)
    - Inserting binary data as FILE
    - NULL handling with FILE
    - Bulk insert with FILE domain
    - Exact matches on FILE columns
    - octet_length() function on FILE
    - FILE/BYTEA compatibility and casting
    """
    assertions = Assertions(db_conn)

    try:
        # Create table with FILE domain columns
        await db_conn.execute("""
            CREATE TABLE file_test (
                id INT,
                document FILE,
                attachment FILE
            ) USING deeplake
        """)

        # Test basic inserts with FILE domain (explicit cast)
        await db_conn.execute("""
            INSERT INTO file_test (id, document, attachment)
            VALUES
                (1, '\\x48656c6c6f'::FILE, '\\x576f726c64'::FILE),
                (2, '\\xDEADBEEF'::FILE, '\\xCAFEBABE'::FILE)
        """)

        # Test inserts without explicit cast (BYTEA is implicitly accepted)
        await db_conn.execute("""
            INSERT INTO file_test (id, document, attachment)
            VALUES
                (3, '\\x00010203', '\\x04050607')
        """)

        # Test NULL handling
        await db_conn.execute("""
            INSERT INTO file_test (id, document, attachment)
            VALUES
                (4, NULL, '\\x08090A0B'::FILE),
                (5, '\\x0C0D0E0F'::FILE, NULL),
                (6, NULL, NULL)
        """)

        # Test bulk insert with FILE domain
        await db_conn.execute("""
            INSERT INTO file_test (id, document, attachment)
            SELECT
                i + 10,
                ('\\x' || lpad(to_hex(i), 8, '0'))::FILE,
                ('\\x' || lpad(to_hex(i * 2), 8, '0'))::FILE
            FROM generate_series(1, 5) AS i
        """)

        # Test exact matches for FILE columns
        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM file_test WHERE document = FROM_HEX('48656c6c6f')"
        )

        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM file_test WHERE attachment = FROM_HEX('576f726c64')"
        )

        # Test NULL filtering
        await assertions.assert_query_row_count(
            2,
            "SELECT * FROM file_test WHERE document IS NULL"
        )

        await assertions.assert_query_row_count(
            2,
            "SELECT * FROM file_test WHERE attachment IS NULL"
        )

        await assertions.assert_query_row_count(
            9,
            "SELECT * FROM file_test WHERE document IS NOT NULL"
        )

        await assertions.assert_query_row_count(
            9,
            "SELECT * FROM file_test WHERE attachment IS NOT NULL"
        )

        # Test that FILE behaves like BYTEA - octet_length function should work
        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM file_test WHERE octet_length(document) = 5 AND id = 1"
        )

        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM file_test WHERE octet_length(document) = 4 AND id = 2"
        )

        # Test that FILE and BYTEA are compatible (can compare/cast)
        await db_conn.execute("SET pg_deeplake.use_deeplake_executor = off")

        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM file_test WHERE document = FROM_HEX('48656c6c6f')::BYTEA AND id = 1"
        )

        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM file_test WHERE document::BYTEA = FROM_HEX('48656c6c6f')::BYTEA AND id = 1"
        )

        await db_conn.execute("RESET pg_deeplake.use_deeplake_executor")

        print("✓ Test passed: FILE domain type works correctly")

    finally:
        # Cleanup
        await db_conn.execute("DROP TABLE IF EXISTS file_test CASCADE")
        await db_conn.execute("RESET pg_deeplake.use_deeplake_executor")


@pytest.mark.asyncio
async def test_file_domain_alter_table(db_conn: asyncpg.Connection):
    """
    Test ALTER TABLE ADD COLUMN with FILE domain type.
    """
    assertions = Assertions(db_conn)

    try:
        # Create initial table
        await db_conn.execute("""
            CREATE TABLE file_alter_test (
                id INT,
                name TEXT
            ) USING deeplake
        """)

        # Insert initial data
        await db_conn.execute("""
            INSERT INTO file_alter_test (id, name)
            VALUES (1, 'row1'), (2, 'row2'), (3, 'row3')
        """)

        # Add FILE column
        await db_conn.execute("""
            ALTER TABLE file_alter_test ADD COLUMN file_col FILE
        """)

        # Update with FILE data
        await db_conn.execute("""
            UPDATE file_alter_test SET file_col = '\\x48656c6c6f'::FILE WHERE id = 1
        """)
        await db_conn.execute("""
            UPDATE file_alter_test SET file_col = '\\x576f726c64'::FILE WHERE id = 2
        """)

        # Verify
        await assertions.assert_query_row_count(
            2,
            "SELECT * FROM file_alter_test WHERE file_col IS NOT NULL"
        )

        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM file_alter_test WHERE file_col IS NULL"
        )

        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM file_alter_test WHERE file_col = FROM_HEX('48656c6c6f')"
        )

        print("✓ Test passed: ALTER TABLE ADD COLUMN with FILE domain works correctly")

    finally:
        await db_conn.execute("DROP TABLE IF EXISTS file_alter_test CASCADE")
