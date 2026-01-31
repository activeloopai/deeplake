"""
Test VIDEO domain type (domain over BYTEA, maps to video_type in deeplake).
"""
import pytest
import asyncpg
from test_utils.assertions import Assertions


@pytest.mark.asyncio
@pytest.mark.skip
async def test_video_domain(db_conn: asyncpg.Connection):
    """
    Test VIDEO domain type for video data.

    Tests:
    - Creating VIDEO columns (domain over BYTEA)
    - Inserting binary data as VIDEO
    - NULL handling with VIDEO
    - Bulk insert with VIDEO domain
    - Exact matches on VIDEO columns
    - octet_length() function on VIDEO
    - VIDEO/BYTEA compatibility and casting
    """
    assertions = Assertions(db_conn)

    try:
        # Create table with VIDEO domain columns
        await db_conn.execute("""
            CREATE TABLE video_test (
                id INT,
                main_video VIDEO,
                preview VIDEO
            ) USING deeplake
        """)

        # Test basic inserts with VIDEO domain (explicit cast)
        await db_conn.execute("""
            INSERT INTO video_test (id, main_video, preview)
            VALUES
                (1, '\\x000000186674797069736f6d'::VIDEO, '\\x1a45dfa3'::VIDEO),
                (2, '\\x52494646'::VIDEO, '\\x4f676753'::VIDEO)
        """)

        # Test inserts without explicit cast (BYTEA is implicitly accepted)
        await db_conn.execute("""
            INSERT INTO video_test (id, main_video, preview)
            VALUES
                (3, '\\x00010203', '\\x04050607')
        """)

        # Test NULL handling
        await db_conn.execute("""
            INSERT INTO video_test (id, main_video, preview)
            VALUES
                (4, NULL, '\\x08090A0B'::VIDEO),
                (5, '\\x0C0D0E0F'::VIDEO, NULL),
                (6, NULL, NULL)
        """)

        # Test bulk insert with VIDEO domain
        await db_conn.execute("""
            INSERT INTO video_test (id, main_video, preview)
            SELECT
                i + 10,
                ('\\x' || lpad(to_hex(i), 8, '0'))::VIDEO,
                ('\\x' || lpad(to_hex(i * 2), 8, '0'))::VIDEO
            FROM generate_series(1, 5) AS i
        """)

        # Test exact matches for VIDEO columns
        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM video_test WHERE main_video = FROM_HEX('000000186674797069736f6d')"
        )

        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM video_test WHERE preview = FROM_HEX('1a45dfa3')"
        )

        # Test NULL filtering
        await assertions.assert_query_row_count(
            2,
            "SELECT * FROM video_test WHERE main_video IS NULL"
        )

        await assertions.assert_query_row_count(
            2,
            "SELECT * FROM video_test WHERE preview IS NULL"
        )

        await assertions.assert_query_row_count(
            9,
            "SELECT * FROM video_test WHERE main_video IS NOT NULL"
        )

        await assertions.assert_query_row_count(
            9,
            "SELECT * FROM video_test WHERE preview IS NOT NULL"
        )

        # Test that VIDEO behaves like BYTEA - octet_length function should work
        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM video_test WHERE octet_length(main_video) = 12 AND id = 1"
        )

        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM video_test WHERE octet_length(main_video) = 4 AND id = 2"
        )

        # Test that VIDEO and BYTEA are compatible (can compare/cast)
        await db_conn.execute("SET pg_deeplake.use_deeplake_executor = off")

        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM video_test WHERE main_video = FROM_HEX('52494646')::BYTEA AND id = 2"
        )

        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM video_test WHERE main_video::BYTEA = FROM_HEX('52494646')::BYTEA AND id = 2"
        )

        await db_conn.execute("RESET pg_deeplake.use_deeplake_executor")

        print("✓ Test passed: VIDEO domain type works correctly")

    finally:
        # Cleanup
        await db_conn.execute("DROP TABLE IF EXISTS video_test CASCADE")
        await db_conn.execute("RESET pg_deeplake.use_deeplake_executor")


@pytest.mark.asyncio
@pytest.mark.skip
async def test_video_domain_alter_table(db_conn: asyncpg.Connection):
    """
    Test ALTER TABLE ADD COLUMN with VIDEO domain type.
    """
    assertions = Assertions(db_conn)

    try:
        # Create initial table
        await db_conn.execute("""
            CREATE TABLE video_alter_test (
                id INT,
                title TEXT
            ) USING deeplake
        """)

        # Insert initial data
        await db_conn.execute("""
            INSERT INTO video_alter_test (id, title)
            VALUES (1, 'video1'), (2, 'video2'), (3, 'video3')
        """)

        # Add VIDEO column
        await db_conn.execute("""
            ALTER TABLE video_alter_test ADD COLUMN video_col VIDEO
        """)

        # Update with VIDEO data
        await db_conn.execute("""
            UPDATE video_alter_test SET video_col = '\\x000000186674797069736f6d'::VIDEO WHERE id = 1
        """)
        await db_conn.execute("""
            UPDATE video_alter_test SET video_col = '\\x1a45dfa3'::VIDEO WHERE id = 2
        """)

        # Verify
        await assertions.assert_query_row_count(
            2,
            "SELECT * FROM video_alter_test WHERE video_col IS NOT NULL"
        )

        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM video_alter_test WHERE video_col IS NULL"
        )

        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM video_alter_test WHERE video_col = FROM_HEX('000000186674797069736f6d')"
        )

        print("✓ Test passed: ALTER TABLE ADD COLUMN with VIDEO domain works correctly")

    finally:
        await db_conn.execute("DROP TABLE IF EXISTS video_alter_test CASCADE")


@pytest.mark.asyncio
@pytest.mark.skip
async def test_mixed_domain_types(db_conn: asyncpg.Connection):
    """
    Test table with multiple domain types: FILE, IMAGE, VIDEO, and BYTEA.
    """
    assertions = Assertions(db_conn)

    try:
        # Create table with all domain types
        await db_conn.execute("""
            CREATE TABLE mixed_domain_test (
                id INT,
                file_data FILE,
                image_data IMAGE,
                video_data VIDEO,
                raw_data BYTEA
            ) USING deeplake
        """)

        # Insert data with explicit casts
        await db_conn.execute("""
            INSERT INTO mixed_domain_test (id, file_data, image_data, video_data, raw_data)
            VALUES
                (1, '\\x48656c6c6f'::FILE, '\\x89504e47'::IMAGE, '\\x1a45dfa3'::VIDEO, '\\xDEADBEEF'::BYTEA)
        """)

        # Insert data without explicit casts (BYTEA implicitly accepted for all domain types)
        await db_conn.execute("""
            INSERT INTO mixed_domain_test (id, file_data, image_data, video_data, raw_data)
            VALUES
                (2, '\\x576f726c64', '\\xffd8ffe0', '\\x52494646', '\\xCAFEBABE'),
                (3, NULL, NULL, NULL, NULL)
        """)

        # Verify row count
        await assertions.assert_query_row_count(
            3,
            "SELECT * FROM mixed_domain_test"
        )

        # Verify each column type works
        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM mixed_domain_test WHERE file_data = FROM_HEX('48656c6c6f')"
        )

        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM mixed_domain_test WHERE image_data = FROM_HEX('89504e47')"
        )

        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM mixed_domain_test WHERE video_data = FROM_HEX('1a45dfa3')"
        )

        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM mixed_domain_test WHERE raw_data = FROM_HEX('DEADBEEF')"
        )

        # Test NULL filtering for each type
        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM mixed_domain_test WHERE file_data IS NULL"
        )

        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM mixed_domain_test WHERE image_data IS NULL"
        )

        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM mixed_domain_test WHERE video_data IS NULL"
        )

        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM mixed_domain_test WHERE raw_data IS NULL"
        )

        print("✓ Test passed: Mixed domain types (FILE, IMAGE, VIDEO, BYTEA) work correctly")

    finally:
        await db_conn.execute("DROP TABLE IF EXISTS mixed_domain_test CASCADE")
