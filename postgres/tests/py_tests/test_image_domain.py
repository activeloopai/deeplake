"""
Test IMAGE domain type (domain over BYTEA).

Ported from: postgres/tests/sql/image_domain.sql
"""
import pytest
import asyncpg
from test_utils.assertions import Assertions


@pytest.mark.asyncio
async def test_image_domain(db_conn: asyncpg.Connection):
    """
    Test IMAGE domain type for binary image data.

    Tests:
    - Creating IMAGE columns (domain over BYTEA)
    - Inserting binary data as IMAGE
    - NULL handling with IMAGE
    - Bulk insert with IMAGE domain
    - Exact matches on IMAGE columns
    - octet_length() function on IMAGE
    - Comparison operators on IMAGE
    - IMAGE/BYTEA compatibility and casting
    """
    assertions = Assertions(db_conn)

    try:
        # Create table with IMAGE domain columns
        await db_conn.execute("""
            CREATE TABLE image_test (
                id INT,
                photo IMAGE,
                thumbnail IMAGE
            ) USING deeplake
        """)

        # Test basic inserts with IMAGE domain
        await db_conn.execute("""
            INSERT INTO image_test (id, photo, thumbnail)
            VALUES
                (1, '\\x89504E47'::IMAGE, '\\xFF'::IMAGE),
                (2, '\\xDEADBEEF'::IMAGE, '\\xCAFEBABE'::IMAGE),
                (3, '\\x00010203'::IMAGE, '\\x04050607'::IMAGE)
        """)

        # Test NULL handling
        await db_conn.execute("""
            INSERT INTO image_test (id, photo, thumbnail)
            VALUES
                (4, NULL, '\\x08090A0B'::IMAGE),
                (5, '\\x0C0D0E0F'::IMAGE, NULL),
                (6, NULL, NULL)
        """)

        # Test bulk insert with IMAGE domain
        await db_conn.execute("""
            INSERT INTO image_test (id, photo, thumbnail)
            SELECT
                i + 10,
                ('\\x' || lpad(to_hex(i), 8, '0'))::IMAGE,
                ('\\x' || lpad(to_hex(i * 2), 8, '0'))::IMAGE
            FROM generate_series(1, 5) AS i
        """)

        # Test exact matches for IMAGE columns
        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM image_test WHERE photo = FROM_HEX('89504E47')"
        )

        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM image_test WHERE thumbnail = FROM_HEX('FF')"
        )

        # Test NULL filtering
        await assertions.assert_query_row_count(
            2,
            "SELECT * FROM image_test WHERE photo IS NULL"
        )

        await assertions.assert_query_row_count(
            2,
            "SELECT * FROM image_test WHERE thumbnail IS NULL"
        )

        await assertions.assert_query_row_count(
            9,
            "SELECT * FROM image_test WHERE photo IS NOT NULL"
        )

        await assertions.assert_query_row_count(
            9,
            "SELECT * FROM image_test WHERE thumbnail IS NOT NULL"
        )

        # Test that IMAGE behaves like BYTEA - octet_length function should work
        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM image_test WHERE octet_length(photo) = 4 AND id = 1"
        )

        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM image_test WHERE octet_length(photo) = 4 AND id = 2"
        )

        # Test comparison operators
        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM image_test WHERE photo > FROM_HEX('00000000') AND id = 1"
        )

        # Test that IMAGE and BYTEA are compatible (can compare/cast)
        await db_conn.execute("SET pg_deeplake.use_deeplake_executor = off")

        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM image_test WHERE photo = FROM_HEX('89504E47')::BYTEA AND id = 1"
        )

        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM image_test WHERE photo::BYTEA = FROM_HEX('89504E47')::BYTEA AND id = 1"
        )

        await db_conn.execute("RESET pg_deeplake.use_deeplake_executor")

        print("✓ Test passed: IMAGE domain type works correctly")

    finally:
        # Cleanup
        await db_conn.execute("DROP TABLE IF EXISTS image_test CASCADE")
        await db_conn.execute("RESET pg_deeplake.use_deeplake_executor")
