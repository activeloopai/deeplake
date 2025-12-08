"""
Test date and time data types with deeplake storage.

Ported from: postgres/tests/sql/date_time.sql
"""
import pytest
import asyncpg
from datetime import date, time, datetime
from lib.assertions import Assertions


@pytest.mark.asyncio
@pytest.mark.slow
async def test_date_time_types(db_conn: asyncpg.Connection):
    """
    Test date/time operations with deeplake storage.

    Tests:
    - date, time, timestamp, timestamptz types
    - Exact matching for each date/time type
    - Comparison operators (<, >, <=, >=)
    - Bulk insert with date/time arithmetic (10k rows)
    """
    assertions = Assertions(db_conn)

    try:
        # Create table with date/time columns
        await db_conn.execute("""
            CREATE TABLE people (
                name text,
                last_name text,
                age int,
                birth_date date,
                birth_time time,
                created_at timestamp,
                created_at_tz timestamptz
            ) USING deeplake
        """)

        # Insert initial rows with various date/time values
        await db_conn.execute("""
            INSERT INTO people (name, last_name, age, birth_date, birth_time, created_at, created_at_tz)
            VALUES
                ('n1', 'l1', 1, '2000-01-01', '08:00:00', '2020-01-01 10:00:00', '2020-01-01 10:00:00+00'),
                ('n2', 'l2', 2, '2001-02-02', '09:30:00', '2021-02-02 11:30:00', '2021-02-02 11:30:00+01'),
                ('n3', 'l3', 3, '2002-03-03', '10:45:00', '2022-03-03 12:45:00', '2022-03-03 12:45:00-05'),
                ('n4', 'l4', 4, '2003-04-04', '11:15:00', '2023-04-04 13:15:00', '2023-04-04 13:15:00+03')
        """)

        await assertions.assert_table_row_count(4, "people")

        # Additional inserts including duplicate
        await db_conn.execute("""
            INSERT INTO people (name, last_name, age, birth_date, birth_time, created_at, created_at_tz)
            VALUES
                ('n4', 'l4', 4, '2003-04-04', '11:15:00', '2023-04-04 13:15:00', '2023-04-04 13:15:00+03'),
                ('n5', 'l5', 5, '2004-05-05', '12:00:00', '2024-05-05 14:00:00', '2024-05-05 14:00:00+02'),
                ('n6', 'l6', 6, '2005-06-06', '13:30:00', '2025-06-06 15:30:00', '2025-06-06 15:30:00-07')
        """)

        await assertions.assert_table_row_count(7, "people")

        # Test exact matches for each date/time type
        await assertions.assert_query_row_count(
            2,
            "SELECT * FROM people WHERE age = 4"
        )

        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM people WHERE birth_date = '2001-02-02'"
        )

        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM people WHERE birth_time = '09:30:00'"
        )

        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM people WHERE created_at = '2021-02-02 11:30:00'"
        )

        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM people WHERE created_at_tz = '2021-02-02 11:30:00+01'"
        )

        # Test date/time comparison operators
        await assertions.assert_query_row_count(
            5,
            "SELECT * FROM people WHERE birth_date > '2002-01-01'"
        )

        await assertions.assert_query_row_count(
            2,
            "SELECT * FROM people WHERE birth_time < '10:00:00'"
        )

        await assertions.assert_query_row_count(
            5,
            "SELECT * FROM people WHERE created_at >= '2022-01-01 00:00:00'"
        )

        await assertions.assert_query_row_count(
            6,
            "SELECT * FROM people WHERE created_at_tz <= '2024-12-31 23:59:59+00'"
        )

        # Bulk insert with date/time arithmetic (10k rows)
        await db_conn.execute("""
            INSERT INTO people
            SELECT
                'n'||i,
                'l'||i,
                i,
                '2000-01-01'::date + (i || ' days')::interval,
                '08:00:00'::time + (i || ' minutes')::interval,
                '2020-01-01 10:00:00'::timestamp + (i || ' hours')::interval,
                '2020-01-01 10:00:00+00'::timestamptz + (i || ' hours')::interval
            FROM generate_series(1, 10000) i
        """)

        await assertions.assert_table_row_count(10007, "people")

        print("✓ Test passed: Date/time types work correctly with deeplake storage")

    finally:
        # Cleanup
        await db_conn.execute("DROP TABLE IF EXISTS people CASCADE")
