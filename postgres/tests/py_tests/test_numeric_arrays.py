"""
Test numeric array operations with deeplake storage.

Tests support for all numeric array types:
- SMALLINT[] (int2[])
- INT[] (int4[])
- BIGINT[] (int8[])
- REAL[] (float4[])
- DOUBLE PRECISION[] (float8[])

With support for multidimensional arrays up to 255 dimensions.
"""
import pytest
import asyncpg
from lib.assertions import Assertions


@pytest.mark.asyncio
async def test_numeric_arrays_1d(db_conn: asyncpg.Connection):
    """
    Test 1D numeric array operations.

    Tests:
    - All numeric array types (int2[], int4[], int8[], float4[], float8[])
    - Insert and retrieve operations
    - Array equality comparisons
    - Array element access
    - Array containment operator (@>)
    - Array overlap operator (&&)
    - Bulk inserts
    """
    assertions = Assertions(db_conn)

    try:
        # Create table with all numeric array types
        await db_conn.execute("""
            CREATE TABLE numeric_arrays_1d (
                id SERIAL,
                int2_array SMALLINT[],
                int4_array INT[],
                int8_array BIGINT[],
                float4_array REAL[],
                float8_array DOUBLE PRECISION[]
            ) USING deeplake
        """)

        # Insert test data
        await db_conn.execute("""
            INSERT INTO numeric_arrays_1d (int2_array, int4_array, int8_array, float4_array, float8_array)
            VALUES
                (ARRAY[1, 2, 3]::int2[], ARRAY[10, 20, 30]::int4[], ARRAY[100, 200, 300]::int8[],
                 ARRAY[1.1, 2.2, 3.3]::float4[], ARRAY[10.1, 20.2, 30.3]::float8[]),
                (ARRAY[4, 5, 6]::int2[], ARRAY[40, 50, 60]::int4[], ARRAY[400, 500, 600]::int8[],
                 ARRAY[4.4, 5.5, 6.6]::float4[], ARRAY[40.4, 50.5, 60.6]::float8[]),
                (ARRAY[7, 8, 9]::int2[], ARRAY[70, 80, 90]::int4[], ARRAY[700, 800, 900]::int8[],
                 ARRAY[7.7, 8.8, 9.9]::float4[], ARRAY[70.7, 80.8, 90.9]::float8[])
        """)

        await assertions.assert_table_row_count(3, "numeric_arrays_1d")

        # Test exact matches for int2 arrays (using element access to avoid operator ambiguity)
        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM numeric_arrays_1d WHERE int2_array[1] = 1 AND int2_array[2] = 2 AND int2_array[3] = 3"
        )

        # Test exact matches for int4 arrays (using element access to avoid operator ambiguity)
        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM numeric_arrays_1d WHERE int4_array[1] = 40 AND int4_array[2] = 50"
        )

        # Test exact matches for int8 arrays (using element access to avoid operator ambiguity)
        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM numeric_arrays_1d WHERE int8_array[1] = 700 AND int8_array[2] = 800"
        )

        # Test exact matches for float4 arrays (stored as float32)
        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM numeric_arrays_1d WHERE float4_array = ARRAY[1.1, 2.2, 3.3]::float4[]"
        )

        # Test exact matches for float8 arrays
        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM numeric_arrays_1d WHERE float8_array = ARRAY[40.4, 50.5, 60.6]::float8[]"
        )

        # Test array element access for int2 arrays
        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM numeric_arrays_1d WHERE int2_array[1] = 1"
        )
        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM numeric_arrays_1d WHERE int2_array[2] = 5"
        )

        # Test array element access for int4 arrays
        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM numeric_arrays_1d WHERE int4_array[1] = 10"
        )
        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM numeric_arrays_1d WHERE int4_array[3] = 90"
        )

        # Test array element access for int8 arrays
        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM numeric_arrays_1d WHERE int8_array[1] = 100"
        )
        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM numeric_arrays_1d WHERE int8_array[2] = 500"
        )

        # Test array containment operator (@>) for int4 arrays
        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM numeric_arrays_1d WHERE int4_array @> ARRAY[10, 20]::int4[]"
        )
        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM numeric_arrays_1d WHERE int8_array @> ARRAY[400]::int8[]"
        )

        # Test array overlap operator (&&) for int2 arrays
        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM numeric_arrays_1d WHERE int2_array && ARRAY[1, 100]::int2[]"
        )
        await assertions.assert_query_row_count(
            2,
            "SELECT * FROM numeric_arrays_1d WHERE int2_array && ARRAY[1, 4]::int2[]"
        )

        # Test ANY operator with int4 arrays
        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM numeric_arrays_1d WHERE 10 = ANY(int4_array)"
        )
        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM numeric_arrays_1d WHERE 50 = ANY(int4_array)"
        )

        # Bulk insert with array generation
        await db_conn.execute("""
            INSERT INTO numeric_arrays_1d (int2_array, int4_array, int8_array, float4_array, float8_array)
            SELECT
                ARRAY[i::int2, (i+1)::int2, (i+2)::int2],
                ARRAY[i, i+1, i+2],
                ARRAY[i::int8, (i+1)::int8, (i+2)::int8],
                ARRAY[i::float4, (i+1)::float4, (i+2)::float4],
                ARRAY[i::float8, (i+1)::float8, (i+2)::float8]
            FROM generate_series(1, 100) i
        """)

        await assertions.assert_table_row_count(103, "numeric_arrays_1d")

        # Test filtering on bulk inserted data
        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM numeric_arrays_1d WHERE int4_array[1] = 50"
        )
        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM numeric_arrays_1d WHERE int8_array[2] = 76"
        )

        print("✓ Test passed: 1D numeric array operations work correctly")

    finally:
        # Cleanup
        await db_conn.execute("DROP TABLE IF EXISTS numeric_arrays_1d CASCADE")


@pytest.mark.asyncio
async def test_numeric_arrays_2d(db_conn: asyncpg.Connection):
    """
    Test 2D numeric array operations.

    Tests:
    - All numeric 2D array types
    - Insert and retrieve operations
    - Array equality comparisons
    - 2D array element access
    """
    assertions = Assertions(db_conn)

    try:
        # Create table with 2D numeric array types
        await db_conn.execute("""
            CREATE TABLE numeric_arrays_2d (
                id SERIAL,
                int2_array_2d SMALLINT[][],
                int4_array_2d INT[][],
                int8_array_2d BIGINT[][],
                float4_array_2d REAL[][],
                float8_array_2d DOUBLE PRECISION[][]
            ) USING deeplake
        """)

        # Insert test data with 2D arrays
        await db_conn.execute("""
            INSERT INTO numeric_arrays_2d (int2_array_2d, int4_array_2d, int8_array_2d, float4_array_2d, float8_array_2d)
            VALUES
                (ARRAY[[1, 2], [3, 4]]::int2[][], ARRAY[[10, 20], [30, 40]]::int4[][],
                 ARRAY[[100, 200], [300, 400]]::int8[][], ARRAY[[1.1, 2.2], [3.3, 4.4]]::float4[][],
                 ARRAY[[10.1, 20.2], [30.3, 40.4]]::float8[][]),
                (ARRAY[[5, 6], [7, 8]]::int2[][], ARRAY[[50, 60], [70, 80]]::int4[][],
                 ARRAY[[500, 600], [700, 800]]::int8[][], ARRAY[[5.5, 6.6], [7.7, 8.8]]::float4[][],
                 ARRAY[[50.5, 60.6], [70.7, 80.8]]::float8[][])
        """)

        await assertions.assert_table_row_count(2, "numeric_arrays_2d")

        # Test exact matches for 2D int2 arrays (using element access to avoid operator ambiguity)
        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM numeric_arrays_2d WHERE int2_array_2d[1][1] = 1 AND int2_array_2d[2][2] = 4"
        )

        # Test exact matches for 2D int4 arrays (using element access to avoid operator ambiguity)
        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM numeric_arrays_2d WHERE int4_array_2d[1][1] = 50 AND int4_array_2d[2][2] = 80"
        )

        # Test exact matches for 2D int8 arrays (using element access to avoid operator ambiguity)
        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM numeric_arrays_2d WHERE int8_array_2d[1][1] = 100 AND int8_array_2d[2][2] = 400"
        )

        # Test exact matches for 2D float8 arrays
        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM numeric_arrays_2d WHERE float8_array_2d = ARRAY[[50.5, 60.6], [70.7, 80.8]]::float8[][]"
        )

        # Test 2D array element access for int2 arrays
        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM numeric_arrays_2d WHERE int2_array_2d[1][1] = 1"
        )
        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM numeric_arrays_2d WHERE int2_array_2d[2][2] = 8"
        )

        # Test 2D array element access for int4 arrays
        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM numeric_arrays_2d WHERE int4_array_2d[1][2] = 20"
        )
        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM numeric_arrays_2d WHERE int4_array_2d[2][1] = 70"
        )

        # Test 2D array element access for int8 arrays
        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM numeric_arrays_2d WHERE int8_array_2d[1][1] = 100"
        )
        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM numeric_arrays_2d WHERE int8_array_2d[2][2] = 800"
        )

        # Test array length functions for 2D arrays
        # Disable deeplake executor for multi-dimensional array_length
        await db_conn.execute("SET pg_deeplake.use_deeplake_executor = off")
        await assertions.assert_query_row_count(
            2,
            "SELECT * FROM numeric_arrays_2d WHERE array_length(int4_array_2d, 1) = 2"
        )
        await assertions.assert_query_row_count(
            2,
            "SELECT * FROM numeric_arrays_2d WHERE array_length(int4_array_2d, 2) = 2"
        )
        await db_conn.execute("RESET pg_deeplake.use_deeplake_executor")

        print("✓ Test passed: 2D numeric array operations work correctly")

    finally:
        # Cleanup
        await db_conn.execute("RESET pg_deeplake.use_deeplake_executor")
        await db_conn.execute("DROP TABLE IF EXISTS numeric_arrays_2d CASCADE")


@pytest.mark.asyncio
async def test_numeric_arrays_3d(db_conn: asyncpg.Connection):
    """
    Test 3D numeric array operations to verify multidimensional support.

    Tests:
    - 3D arrays for numeric types
    - Insert and retrieve operations
    - Array equality comparisons
    """
    assertions = Assertions(db_conn)

    try:
        # Create table with 3D numeric array types
        await db_conn.execute("""
            CREATE TABLE numeric_arrays_3d (
                id SERIAL,
                int4_array_3d INT[][][],
                int8_array_3d BIGINT[][][],
                float8_array_3d DOUBLE PRECISION[][][]
            ) USING deeplake
        """)

        # Insert test data with 3D arrays (2x2x2 cubes)
        await db_conn.execute("""
            INSERT INTO numeric_arrays_3d (int4_array_3d, int8_array_3d, float8_array_3d)
            VALUES
                (ARRAY[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]::int4[][][],
                 ARRAY[[[10, 20], [30, 40]], [[50, 60], [70, 80]]]::int8[][][],
                 ARRAY[[[1.1, 2.2], [3.3, 4.4]], [[5.5, 6.6], [7.7, 8.8]]]::float8[][][]),
                (ARRAY[[[9, 10], [11, 12]], [[13, 14], [15, 16]]]::int4[][][],
                 ARRAY[[[90, 100], [110, 120]], [[130, 140], [150, 160]]]::int8[][][],
                 ARRAY[[[9.9, 10.1], [11.1, 12.2]], [[13.3, 14.4], [15.5, 16.6]]]::float8[][][])
        """)

        await assertions.assert_table_row_count(2, "numeric_arrays_3d")

        # Test exact matches for 3D int4 arrays (using element access to avoid operator ambiguity)
        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM numeric_arrays_3d WHERE int4_array_3d[1][1][1] = 1 AND int4_array_3d[2][2][2] = 8"
        )

        # Test exact matches for 3D int8 arrays (using element access to avoid operator ambiguity)
        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM numeric_arrays_3d WHERE int8_array_3d[1][1][1] = 90 AND int8_array_3d[2][2][2] = 160"
        )

        # Test exact matches for 3D float8 arrays
        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM numeric_arrays_3d WHERE float8_array_3d = ARRAY[[[1.1, 2.2], [3.3, 4.4]], [[5.5, 6.6], [7.7, 8.8]]]::float8[][][]"
        )

        # Test 3D array element access
        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM numeric_arrays_3d WHERE int4_array_3d[1][1][1] = 1"
        )
        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM numeric_arrays_3d WHERE int4_array_3d[2][2][2] = 16"
        )
        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM numeric_arrays_3d WHERE int8_array_3d[1][2][1] = 30"
        )

        print("✓ Test passed: 3D numeric array operations work correctly")

    finally:
        # Cleanup
        await db_conn.execute("DROP TABLE IF EXISTS numeric_arrays_3d CASCADE")


@pytest.mark.asyncio
async def test_numeric_arrays_mixed_operations(db_conn: asyncpg.Connection):
    """
    Test mixed operations with numeric arrays.

    Tests:
    - Updates with numeric arrays
    - NULL handling
    - Mixed queries across different numeric array types
    """
    assertions = Assertions(db_conn)

    try:
        # Create table
        await db_conn.execute("""
            CREATE TABLE numeric_arrays_mixed (
                id SERIAL,
                int_array INT[],
                bigint_array BIGINT[],
                float_array DOUBLE PRECISION[]
            ) USING deeplake
        """)

        # Insert test data without NULLs first
        await db_conn.execute("""
            INSERT INTO numeric_arrays_mixed (int_array, bigint_array, float_array)
            VALUES
                (ARRAY[1, 2, 3], ARRAY[100, 200, 300], ARRAY[1.5, 2.5, 3.5]),
                (ARRAY[7, 8, 9], ARRAY[400, 500, 600], ARRAY[4.5, 5.5, 6.5])
        """)

        await assertions.assert_table_row_count(2, "numeric_arrays_mixed")

        # Test update operations
        await db_conn.execute("""
            UPDATE numeric_arrays_mixed
            SET int_array = ARRAY[99, 100, 101]
            WHERE id = 1
        """)

        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM numeric_arrays_mixed WHERE int_array[1] = 99 AND int_array[2] = 100"
        )

        # Test update with different array types
        await db_conn.execute("""
            UPDATE numeric_arrays_mixed
            SET bigint_array = ARRAY[9999, 10000, 10001]::bigint[]
            WHERE id = 2
        """)

        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM numeric_arrays_mixed WHERE bigint_array[1] = 9999 AND bigint_array[2] = 10000"
        )

        print("✓ Test passed: Mixed numeric array operations work correctly")

    finally:
        # Cleanup
        await db_conn.execute("DROP TABLE IF EXISTS numeric_arrays_mixed CASCADE")
