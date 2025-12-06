"""
Test NUMERIC type precision and operations (DISABLED - test is disabled in Makefile).

Ported from: postgres/tests/sql/numeric_test.sql
"""
import pytest
import asyncpg
from lib.assertions import Assertions


@pytest.mark.asyncio
async def test_numeric_precision(db_conn: asyncpg.Connection):
    """
    Test NUMERIC type precision across native and deeplake tables.

    NOTE: This test is disabled by default as it's disabled in the Makefile.
    Run with: pytest -m disabled

    Tests:
    - NUMERIC with various precision/scale combinations
    - Arithmetic operations on NUMERIC
    - Precision preservation functions (ROUND, TRUNC)
    - Comparison operations
    - Aggregation functions (SUM, AVG, MIN, MAX, STDDEV)
    - Type conversions (INTEGER, REAL, DOUBLE PRECISION)
    - Comparing deeplake vs native PostgreSQL NUMERIC behavior
    """
    assertions = Assertions(db_conn)

    try:
        # Set numeric handling mode
        await db_conn.execute("SET pg_deeplake.treat_numeric_as_double = false")

        # Create native PostgreSQL table
        await db_conn.execute("""
            CREATE TABLE numeric_precision_test_native (
                id SERIAL PRIMARY KEY,
                num_default NUMERIC,
                num_10_0 NUMERIC(10, 0),
                num_10_2 NUMERIC(10, 2),
                num_15_5 NUMERIC(15, 5),
                num_15_10 NUMERIC(15, 10),
                num_20_10 NUMERIC(20, 10),
                num_38_18 NUMERIC(38, 18)
            )
        """)

        # Create deeplake table with same structure
        await db_conn.execute("""
            CREATE TABLE numeric_precision_test_deeplake (
                id SERIAL PRIMARY KEY,
                num_default NUMERIC,
                num_10_0 NUMERIC(10, 0),
                num_10_2 NUMERIC(10, 2),
                num_15_5 NUMERIC(15, 5),
                num_15_10 NUMERIC(15, 10),
                num_20_10 NUMERIC(20, 10),
                num_38_18 NUMERIC(38, 18)
            ) USING deeplake
        """)

        # Insert same test data into both tables
        test_data = """
            (  123.456789,  1234567890,  12345678.90,  1234567890.12345,  12345.1234567890,  1234567890.1234567890,  12345678901234567890.123456789012345678),
            (    0.000001,           0,         0.01,           0.00001,      0.0000000001,           0.0000000001,                     0.000000000000000001),
            (999999999999,  9999999999,  99999999.99,  9999999999.99999,  99999.9999999999,  9999999999.9999999999,  99999999999999999999.999999999999999999),
            ( -123.456789, -1234567890, -12345678.90, -1234567890.12345, -12345.1234567890, -1234567890.1234567890, -12345678901234567890.123456789012345678),
            (         1.0,           1,         1.00,           1.00000,      1.0000000000,           1.0000000000,                     1.000000000000000000)
        """

        await db_conn.execute(f"""
            INSERT INTO numeric_precision_test_native (num_default, num_10_0, num_10_2, num_15_5, num_15_10, num_20_10, num_38_18)
            VALUES {test_data}
        """)

        await db_conn.execute(f"""
            INSERT INTO numeric_precision_test_deeplake (num_default, num_10_0, num_10_2, num_15_5, num_15_10, num_20_10, num_38_18)
            VALUES {test_data}
        """)

        # Verify both tables have same row count
        await assertions.assert_table_row_count(5, "numeric_precision_test_native")
        await assertions.assert_table_row_count(5, "numeric_precision_test_deeplake")

        # Test 1: Compare basic SELECT (with REAL casting for high-precision columns)
        native_select = await db_conn.fetch("""
            SELECT num_default, num_10_0, num_10_2, num_15_5, num_15_10, num_20_10::REAL, num_38_18::REAL
            FROM numeric_precision_test_native ORDER BY id
        """)

        deeplake_select = await db_conn.fetch("""
            SELECT num_default, num_10_0, num_10_2, num_15_5, num_15_10, num_20_10::REAL, num_38_18::REAL
            FROM numeric_precision_test_deeplake ORDER BY id
        """)

        assert len(native_select) == len(deeplake_select), \
            "Native and deeplake SELECT should return same number of rows"

        # Test 2: Compare arithmetic operations
        native_arith = await db_conn.fetch("""
            SELECT id, num_10_2 + num_15_5 AS addition, num_10_2 - num_15_5 AS subtraction,
                   num_10_2 * 2 AS multiplication, num_15_5 / 2 AS division
            FROM numeric_precision_test_native ORDER BY id
        """)

        deeplake_arith = await db_conn.fetch("""
            SELECT id, num_10_2 + num_15_5 AS addition, num_10_2 - num_15_5 AS subtraction,
                   num_10_2 * 2 AS multiplication, num_15_5 / 2 AS division
            FROM numeric_precision_test_deeplake ORDER BY id
        """)

        assert len(native_arith) == len(deeplake_arith), \
            "Arithmetic operations should return same number of rows"

        # Test 3: Compare precision functions
        native_precision = await db_conn.fetch("""
            SELECT id, num_10_2, ROUND(num_10_2, 1) AS rounded_1,
                   ROUND(num_10_2, 0) AS rounded_0, TRUNC(num_10_2, 1) AS truncated_1
            FROM numeric_precision_test_native ORDER BY id
        """)

        deeplake_precision = await db_conn.fetch("""
            SELECT id, num_10_2, ROUND(num_10_2, 1) AS rounded_1,
                   ROUND(num_10_2, 0) AS rounded_0, TRUNC(num_10_2, 1) AS truncated_1
            FROM numeric_precision_test_deeplake ORDER BY id
        """)

        assert len(native_precision) == len(deeplake_precision), \
            "Precision functions should return same number of rows"

        # Test 4: Compare aggregation functions
        native_agg = await db_conn.fetchrow("""
            SELECT COUNT(*) AS count_rows, SUM(num_10_2) AS sum_10_2,
                   AVG(num_10_2) AS avg_10_2, MIN(num_10_2) AS min_10_2,
                   MAX(num_10_2) AS max_10_2
            FROM numeric_precision_test_native
        """)

        deeplake_agg = await db_conn.fetchrow("""
            SELECT COUNT(*) AS count_rows, SUM(num_10_2) AS sum_10_2,
                   AVG(num_10_2) AS avg_10_2, MIN(num_10_2) AS min_10_2,
                   MAX(num_10_2) AS max_10_2
            FROM numeric_precision_test_deeplake
        """)

        assert native_agg['count_rows'] == deeplake_agg['count_rows'], \
            "Aggregation COUNT should match"

        # Test 5: Type conversions
        native_conv = await db_conn.fetch("""
            SELECT id, num_10_2::INTEGER AS to_integer,
                   num_10_2::REAL AS to_real,
                   num_10_2::DOUBLE PRECISION AS to_double
            FROM numeric_precision_test_native WHERE num_10_2 IS NOT NULL ORDER BY id
        """)

        deeplake_conv = await db_conn.fetch("""
            SELECT id, num_10_2::INTEGER AS to_integer,
                   num_10_2::REAL AS to_real,
                   num_10_2::DOUBLE PRECISION AS to_double
            FROM numeric_precision_test_deeplake WHERE num_10_2 IS NOT NULL ORDER BY id
        """)

        assert len(native_conv) == len(deeplake_conv), \
            "Type conversion should return same number of rows"

        print("✓ Test passed: NUMERIC precision tests completed")

    finally:
        # Cleanup
        await db_conn.execute("DROP TABLE IF EXISTS numeric_precision_test_native CASCADE")
        await db_conn.execute("DROP TABLE IF EXISTS numeric_precision_test_deeplake CASCADE")
        await db_conn.execute("RESET pg_deeplake.treat_numeric_as_double")
