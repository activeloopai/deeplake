"""
Test simple data types with deeplake storage.

Ported from: postgres/tests/sql/simple_types.sql
"""
import pytest
import asyncpg
from datetime import datetime, date, time
from lib.assertions import Assertions


@pytest.mark.asyncio
async def test_simple_types(db_conn: asyncpg.Connection):
    """
    Test various simple data types with deeplake storage.

    Tests:
    - boolean, int2, int4, int8
    - float4, float8, numeric
    - date, time, timestamp, timestamptz
    - char, bytea
    - Comparison operators on date/time types
    - Bulk inserts with date/time arithmetic
    """
    assertions = Assertions(db_conn)

    try:
        # Create table with various types
        await db_conn.execute("""
            CREATE TABLE type_test (
                bool_col boolean,
                int2_col int2,
                int4_col int4,
                date_col date,
                time_col time,
                timestamp_col timestamp,
                timestamptz_col timestamptz,
                int8_col int8,
                float4_col float4,
                float8_col float8,
                numeric_col numeric,
                char_col char,
                bytea_col bytea
            ) USING deeplake
        """)

        # Insert test data with different date/time values
        await db_conn.execute("""
            INSERT INTO type_test (
                bool_col, int2_col, int4_col, date_col, time_col,
                timestamp_col, timestamptz_col, int8_col, float4_col,
                float8_col, numeric_col, char_col, bytea_col
            )
            VALUES
                (false, 1, 1, '2000-01-01', '08:00:00', '2020-01-01 10:00:00',
                 '2020-01-01 10:00:00+00', 1, 1.0, 1.0, 1.0, 'a',
                 decode('DEADBEEF', 'hex')),
                (true, 2, 2, '2001-02-02', '09:30:00', '2021-02-02 11:30:00',
                 '2021-02-02 11:30:00+01', 2, 2.0, 2.0, 2.0, 'b',
                 decode('48656c6c6f', 'hex')),
                (false, 3, 3, '2002-03-03', '10:45:00', '2022-03-03 12:45:00',
                 '2022-03-03 12:45:00-05', 3, 3.0, 3.0, 3.0, 'c',
                 decode('DEADBEEF', 'hex')),
                (false, 4, 4, '2003-04-04', '11:15:00', '2023-04-04 13:15:00',
                 '2023-04-04 13:15:00+03', 4, 4.0, 4.0, 4.0, 'd',
                 decode('48656c6c6f', 'hex'))
        """)

        # Verify initial row count
        await assertions.assert_table_row_count(4, "type_test")

        # Insert additional rows including duplicates
        await db_conn.execute("""
            INSERT INTO type_test (
                bool_col, int2_col, int4_col, date_col, time_col,
                timestamp_col, timestamptz_col, int8_col, float4_col,
                float8_col, numeric_col, char_col, bytea_col
            )
            VALUES
                (false, 4, 4, '2003-04-04', '11:15:00', '2023-04-04 13:15:00',
                 '2023-04-04 13:15:00+03', 4, 4.0, 4.0, 4.0, 'd',
                 decode('48656c6c6f', 'hex')),
                (true, 5, 5, '2004-05-05', '12:00:00', '2024-05-05 14:00:00',
                 '2024-05-05 14:00:00+02', 5, 5.0, 5.0, 5.0, 'e',
                 decode('48656c6c6f', 'hex')),
                (false, 6, 6, '2005-06-06', '13:30:00', '2025-06-06 15:30:00',
                 '2025-06-06 15:30:00-07', 6, 6.0, 6.0, 6.0, 'f',
                 decode('DEADBEEF', 'hex'))
        """)

        # Verify total row count
        await assertions.assert_table_row_count(7, "type_test")

        # Test exact matches for each type
        await assertions.assert_query_row_count(
            2,
            "SELECT * FROM type_test WHERE int4_col = 4"
        )
        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM type_test WHERE date_col = '2001-02-02'"
        )
        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM type_test WHERE time_col = '09:30:00'"
        )
        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM type_test WHERE timestamp_col = '2021-02-02 11:30:00'"
        )
        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM type_test WHERE timestamptz_col = '2021-02-02 11:30:00+01'"
        )

        # Test bytea searches (disable deeplake executor for bytea)
        await db_conn.execute("SET pg_deeplake.use_deeplake_executor = off")
        await assertions.assert_query_row_count(
            4,
            "SELECT convert_from(bytea_col, 'UTF8') AS text_val FROM type_test "
            "WHERE POSITION('Hello'::bytea IN bytea_col) > 0"
        )
        await db_conn.execute("RESET pg_deeplake.use_deeplake_executor")

        # Test date/time comparison operators
        await assertions.assert_query_row_count(
            5,
            "SELECT * FROM type_test WHERE date_col > '2002-01-01'"
        )
        await assertions.assert_query_row_count(
            2,
            "SELECT * FROM type_test WHERE time_col < '10:00:00'"
        )
        await assertions.assert_query_row_count(
            5,
            "SELECT * FROM type_test WHERE timestamp_col >= '2022-01-01 00:00:00'"
        )
        await assertions.assert_query_row_count(
            6,
            "SELECT * FROM type_test WHERE timestamptz_col <= '2024-12-31 23:59:59+00'"
        )

        # Bulk insert with date/time arithmetic
        await db_conn.execute("""
            INSERT INTO type_test
            SELECT
                true,
                i,
                i,
                '2000-01-01'::date + (i || ' days')::interval,
                '08:00:00'::time + (i || ' minutes')::interval,
                '2020-01-01 10:00:00'::timestamp + (i || ' hours')::interval,
                '2020-01-01 10:00:00+00'::timestamptz + (i || ' hours')::interval,
                i,
                i::float4,
                i::float8,
                i::numeric,
                chr(97 + (i % 26)),
                decode('48656c6c6f', 'hex')
            FROM generate_series(1, 10000) i
        """)

        # Verify bulk insert
        await assertions.assert_table_row_count(10007, "type_test")

        print("✓ Test passed: All simple types work correctly")

    finally:
        # Cleanup
        await db_conn.execute("RESET pg_deeplake.use_deeplake_executor")
        await db_conn.execute("DROP TABLE IF EXISTS type_test CASCADE")
