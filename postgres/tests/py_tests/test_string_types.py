"""
Test string data types with deeplake storage.

Ported from: postgres/tests/sql/string_types.sql
"""
import pytest
import asyncpg
from test_utils.assertions import Assertions


@pytest.mark.asyncio
async def test_string_types(db_conn: asyncpg.Connection):
    """
    Test various string data types with deeplake storage.

    Tests:
    - char, varchar (with and without length), bpchar (char with length), text
    - Exact matches for string values
    - String comparison operators (>, <, >=, <=)
    - LIKE pattern matching
    - Bulk inserts with string generation (10,000 rows)
    """
    assertions = Assertions(db_conn)

    try:
        # Create table with various string types
        await db_conn.execute("""
            CREATE TABLE type_test (
                char_col char,
                varchar_col varchar(50),
                varchar_1_col varchar(1),
                varchar_generic_col varchar,
                bpchar_col char(13),
                bpchar_1_col char(1),
                text_col text
            ) USING deeplake
        """)

        # Insert test data
        await db_conn.execute("""
            INSERT INTO type_test (
                char_col, varchar_col, varchar_1_col, varchar_generic_col,
                bpchar_col, bpchar_1_col, text_col
            )
            VALUES
                ('a', 'varchar_example_1', 'v', 'varchar_generic_example_1',
                 'bpchar_ex1', 'x', 'text_example_1'),
                ('b', 'varchar_example_2', 'w', 'varchar_generic_example_2',
                 'bpchar_ex2', 'y', 'text_example_2'),
                ('c', 'varchar_example_3', 'x', 'varchar_generic_example_3',
                 'bpchar_ex3', 'z', 'text_example_3'),
                ('d', 'varchar_example_4', 'y', 'varchar_generic_example_4',
                 'bpchar_ex4', 'a', 'text_example_4')
        """)

        await assertions.assert_table_row_count(4, "type_test")

        # Additional inserts including duplicates
        await db_conn.execute("""
            INSERT INTO type_test (
                char_col, varchar_col, varchar_1_col, varchar_generic_col,
                bpchar_col, bpchar_1_col, text_col
            )
            VALUES
                ('d', 'varchar_example_4', 'y', 'varchar_generic_example_4',
                 'bpchar_ex4', 'a', 'text_example_4'),
                ('e', 'varchar_example_5', 'z', 'varchar_generic_example_5',
                 'bpchar_ex5', 'b', 'text_example_5'),
                ('f', 'varchar_example_6', 'v', 'varchar_generic_example_6',
                 'bpchar_ex6', 'c', 'text_example_6')
        """)

        await assertions.assert_table_row_count(7, "type_test")

        # Test exact matches for each string column
        await assertions.assert_query_row_count(
            1, "SELECT * FROM type_test WHERE char_col = 'a'"
        )
        await assertions.assert_query_row_count(
            2, "SELECT * FROM type_test WHERE char_col = 'd'"
        )
        await assertions.assert_query_row_count(
            2, "SELECT * FROM type_test WHERE varchar_col = 'varchar_example_4'"
        )
        await assertions.assert_query_row_count(
            1, "SELECT * FROM type_test WHERE varchar_col = 'varchar_example_1'"
        )
        await assertions.assert_query_row_count(
            2, "SELECT * FROM type_test WHERE varchar_1_col = 'v'"
        )
        await assertions.assert_query_row_count(
            2, "SELECT * FROM type_test WHERE varchar_1_col = 'y'"
        )
        await assertions.assert_query_row_count(
            2, "SELECT * FROM type_test WHERE varchar_generic_col = 'varchar_generic_example_4'"
        )
        await assertions.assert_query_row_count(
            1, "SELECT * FROM type_test WHERE varchar_generic_col = 'varchar_generic_example_5'"
        )
        await assertions.assert_query_row_count(
            2, "SELECT * FROM type_test WHERE bpchar_col = 'bpchar_ex4'"
        )
        await assertions.assert_query_row_count(
            1, "SELECT * FROM type_test WHERE bpchar_col = 'bpchar_ex6'"
        )
        await assertions.assert_query_row_count(
            2, "SELECT * FROM type_test WHERE bpchar_1_col = 'a'"
        )
        await assertions.assert_query_row_count(
            1, "SELECT * FROM type_test WHERE bpchar_1_col = 'c'"
        )
        await assertions.assert_query_row_count(
            2, "SELECT * FROM type_test WHERE text_col = 'text_example_4'"
        )
        await assertions.assert_query_row_count(
            1, "SELECT * FROM type_test WHERE text_col = 'text_example_2'"
        )

        # Test string comparison operators (lexicographic ordering)
        await assertions.assert_query_row_count(
            4, "SELECT * FROM type_test WHERE char_col > 'c'"
        )
        await assertions.assert_query_row_count(
            2, "SELECT * FROM type_test WHERE char_col <= 'b'"
        )
        await assertions.assert_query_row_count(
            4, "SELECT * FROM type_test WHERE varchar_col >= 'varchar_example_4'"
        )
        await assertions.assert_query_row_count(
            3, "SELECT * FROM type_test WHERE varchar_col < 'varchar_example_4'"
        )
        await assertions.assert_query_row_count(
            6, "SELECT * FROM type_test WHERE text_col >= 'text_example_2'"
        )

        # Test LIKE pattern matching
        await assertions.assert_query_row_count(
            7, "SELECT * FROM type_test WHERE varchar_col LIKE 'varchar_example_%'"
        )
        await assertions.assert_query_row_count(
            1, "SELECT * FROM type_test WHERE varchar_col LIKE 'varchar_example_1%'"
        )
        await assertions.assert_query_row_count(
            1, "SELECT * FROM type_test WHERE bpchar_col LIKE 'bpchar_ex1%'"
        )
        await assertions.assert_query_row_count(
            7, "SELECT * FROM type_test WHERE text_col LIKE '%example_%'"
        )

        # Bulk insert with string generation
        await db_conn.execute("""
            INSERT INTO type_test
            SELECT
                chr(97 + (i % 26)),
                'varchar_bulk_' || i,
                chr(97 + (i % 10)),
                'generic_varchar_' || i,
                'bpchar_' || LPAD(i::text, 4, '0'),
                chr(97 + (i % 5)),
                'text_value_' || i
            FROM generate_series(1, 10000) i
        """)

        await assertions.assert_table_row_count(10007, "type_test")

        print("✓ Test passed: All string types work correctly")

    finally:
        # Cleanup
        await db_conn.execute("DROP TABLE IF EXISTS type_test CASCADE")
