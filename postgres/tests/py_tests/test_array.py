"""
Test array operations with deeplake storage.

Ported from: postgres/tests/sql/array.sql
"""
import pytest
import asyncpg
from test_utils.assertions import Assertions


@pytest.mark.asyncio
async def test_array_operations(db_conn: asyncpg.Connection):
    """
    Test various array operations with deeplake storage.

    Tests:
    - 1D and 2D float4 arrays
    - 1D bytea arrays
    - Array equality comparisons
    - Array element access ([1], [1][1])
    - Array containment operator (@>)
    - Array overlap operator (&&)
    - Array length functions
    - ANY operator with arrays
    - Bulk inserts with array generation
    """
    assertions = Assertions(db_conn)

    try:
        # Create table with array columns
        await db_conn.execute("""
            CREATE TABLE array_test (
                float4_array_1d EMBEDDING,
                float4_array_2d EMBEDDING_2D,
                bytea_array_1d bytea[],
                text_array_1d text[],
                varchar_array_1d varchar[]
            ) USING deeplake
        """)

        # Insert test data
        await db_conn.execute("""
            INSERT INTO array_test (float4_array_1d, float4_array_2d, bytea_array_1d, text_array_1d, varchar_array_1d)
            VALUES
                (ARRAY[1.0, 2.0, 3.0]::float4[], ARRAY[[1.0, 2.0], [3.0, 4.0]]::float4[][],
                 ARRAY['\x01'::bytea, '\x02'::bytea, '\x03'::bytea], ARRAY['text1'::text, 'text2'::text, 'text3'::text], ARRAY['varchar1'::varchar, 'varchar2'::varchar, 'varchar3'::varchar]),
                (ARRAY[4.0, 5.0, 6.0]::float4[], ARRAY[[5.0, 6.0], [7.0, 8.0]]::float4[][],
                 ARRAY['\x05'::bytea, '\x06'::bytea, '\x07'::bytea], ARRAY['text4'::text, 'text5'::text, 'text6'::text], ARRAY['varchar4'::varchar, 'varchar5'::varchar, 'varchar6'::varchar]),
                (ARRAY[7.0, 8.0, 9.0]::float4[], ARRAY[[9.0, 10.0], [11.0, 12.0]]::float4[][],
                 ARRAY['\x09'::bytea, '\x0A'::bytea, '\x0B'::bytea], ARRAY['text7'::text, 'text8'::text, 'text9'::text], ARRAY['varchar7'::varchar, 'varchar8'::varchar, 'varchar9'::varchar]),
                (ARRAY[10.0, 11.0, 12.0]::float4[], ARRAY[[13.0, 14.0], [15.0, 16.0]]::float4[][],
                 ARRAY['\x0D'::bytea, '\x0E'::bytea, '\x0F'::bytea], ARRAY['text10'::text, 'text11'::text, 'text12'::text], ARRAY['varchar10'::varchar, 'varchar11'::varchar, 'varchar12'::varchar])
        """)

        await assertions.assert_table_row_count(4, "array_test")

        # Additional inserts including duplicates
        await db_conn.execute("""
            INSERT INTO array_test (float4_array_1d, float4_array_2d, bytea_array_1d, text_array_1d, varchar_array_1d)
            VALUES
                (ARRAY[10.0, 11.0, 12.0]::float4[], ARRAY[[13.0, 14.0], [15.0, 16.0]]::float4[][],
                 ARRAY['\x0D'::bytea, '\x0E'::bytea, '\x0F'::bytea], ARRAY['text13'::text, 'text14'::text, 'text15'::text], ARRAY['varchar13'::varchar, 'varchar14'::varchar, 'varchar15'::varchar]),
                (ARRAY[13.0, 14.0, 15.0]::float4[], ARRAY[[17.0, 18.0], [19.0, 20.0]]::float4[][],
                 ARRAY['\x11'::bytea, '\x12'::bytea, '\x13'::bytea], ARRAY['text16'::text, 'text17'::text, 'text18'::text], ARRAY['varchar16'::varchar, 'varchar17'::varchar, 'varchar18'::varchar]),
                (ARRAY[16.0, 17.0, 18.0]::float4[], ARRAY[[21.0, 22.0], [23.0, 24.0]]::float4[][],
                 ARRAY['\x15'::bytea, '\x16'::bytea, '\x17'::bytea], ARRAY['text19'::text, 'text20'::text, 'text21'::text], ARRAY['varchar19'::varchar, 'varchar20'::varchar, 'varchar21'::varchar])
        """)

        await assertions.assert_table_row_count(7, "array_test")

        # Test exact matches for float4 1D arrays
        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM array_test WHERE float4_array_1d = ARRAY[1.0, 2.0, 3.0]::float4[]"
        )
        await assertions.assert_query_row_count(
            2,
            "SELECT * FROM array_test WHERE float4_array_1d = ARRAY[10.0, 11.0, 12.0]::float4[]"
        )
        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM array_test WHERE float4_array_1d = ARRAY[7.0, 8.0, 9.0]::float4[]"
        )

        # Test exact matches for float4 2D arrays
        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM array_test WHERE float4_array_2d = ARRAY[[1.0, 2.0], [3.0, 4.0]]::float4[][]"
        )
        await assertions.assert_query_row_count(
            2,
            "SELECT * FROM array_test WHERE float4_array_2d = ARRAY[[13.0, 14.0], [15.0, 16.0]]::float4[][]"
        )
        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM array_test WHERE float4_array_2d = ARRAY[[9.0, 10.0], [11.0, 12.0]]::float4[][]"
        )

        # Test exact matches for bytea 1D arrays
        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM array_test WHERE bytea_array_1d = ARRAY['\x01'::bytea, '\x02'::bytea, '\x03'::bytea]"
        )
        await assertions.assert_query_row_count(
            2,
            "SELECT * FROM array_test WHERE bytea_array_1d = ARRAY['\x0D'::bytea, '\x0E'::bytea, '\x0F'::bytea]"
        )
        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM array_test WHERE bytea_array_1d = ARRAY['\x05'::bytea, '\x06'::bytea, '\x07'::bytea]"
        )

        # Test array element access for float4 1D arrays
        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM array_test WHERE float4_array_1d[1] = 1.0"
        )
        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM array_test WHERE float4_array_1d[2] = 5.0"
        )
        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM array_test WHERE float4_array_1d[3] = 9.0"
        )
        await assertions.assert_query_row_count(
            2,
            "SELECT * FROM array_test WHERE float4_array_1d[1] = 10.0"
        )

        # Test array element access for float4 2D arrays
        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM array_test WHERE float4_array_2d[1][1] = 1.0"
        )
        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM array_test WHERE float4_array_2d[2][1] = 7.0"
        )
        await assertions.assert_query_row_count(
            2,
            "SELECT * FROM array_test WHERE float4_array_2d[1][1] = 13.0"
        )

        # Test array element access for bytea 1D arrays
        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM array_test WHERE bytea_array_1d[1] = '\x01'::bytea"
        )
        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM array_test WHERE bytea_array_1d[2] = '\x06'::bytea"
        )
        await assertions.assert_query_row_count(
            2,
            "SELECT * FROM array_test WHERE bytea_array_1d[1] = '\x0D'::bytea"
        )

        # Test array containment operator (@>) for float4 1D arrays
        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM array_test WHERE float4_array_1d @> ARRAY[1.0, 2.0]::float4[]"
        )
        await assertions.assert_query_row_count(
            2,
            "SELECT * FROM array_test WHERE float4_array_1d @> ARRAY[10.0]::float4[]"
        )
        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM array_test WHERE float4_array_1d @> ARRAY[7.0, 8.0, 9.0]::float4[]"
        )

        # Test array overlap operator (&&) for float4 1D arrays
        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM array_test WHERE float4_array_1d && ARRAY[1.0, 100.0]::float4[]"
        )
        await assertions.assert_query_row_count(
            2,
            "SELECT * FROM array_test WHERE float4_array_1d && ARRAY[10.0, 100.0]::float4[]"
        )
        await assertions.assert_query_row_count(
            3,
            "SELECT * FROM array_test WHERE float4_array_1d && ARRAY[1.0, 10.0]::float4[]"
        )

        # Test array length functions for float4 1D arrays
        await assertions.assert_query_row_count(
            7,
            "SELECT * FROM array_test WHERE array_length(float4_array_1d, 1) = 3"
        )
        await assertions.assert_query_row_count(
            0,
            "SELECT * FROM array_test WHERE array_length(float4_array_1d, 1) = 4"
        )

        # Test array length functions for float4 2D arrays
        # Disable deeplake executor for multi-dimensional array_length
        await db_conn.execute("SET pg_deeplake.use_deeplake_executor = off")
        await assertions.assert_query_row_count(
            7,
            "SELECT * FROM array_test WHERE array_length(float4_array_2d, 1) = 2"
        )
        await assertions.assert_query_row_count(
            7,
            "SELECT * FROM array_test WHERE array_length(float4_array_2d, 2) = 2"
        )
        await db_conn.execute("RESET pg_deeplake.use_deeplake_executor")

        # Test array length functions for bytea 1D arrays
        await assertions.assert_query_row_count(
            7,
            "SELECT * FROM array_test WHERE array_length(bytea_array_1d, 1) = 3"
        )

        # Test ANY operator with float4 1D arrays
        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM array_test WHERE 1.0 = ANY(float4_array_1d)"
        )
        await assertions.assert_query_row_count(
            2,
            "SELECT * FROM array_test WHERE 10.0 = ANY(float4_array_1d)"
        )
        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM array_test WHERE 15.0 = ANY(float4_array_1d)"
        )

        # Test array element access for text 1D arrays
        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM array_test WHERE text_array_1d[1] = 'text1'::text"
        )
        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM array_test WHERE text_array_1d[2] = 'text2'::text"
        )
        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM array_test WHERE text_array_1d[3] = 'text3'::text"
        )

        # Test array element access for varchar 1D arrays
        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM array_test WHERE varchar_array_1d[1] = 'varchar1'::varchar"
        )
        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM array_test WHERE varchar_array_1d[2] = 'varchar2'::varchar"
        )
        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM array_test WHERE varchar_array_1d[3] = 'varchar3'::varchar"
        )

        # Bulk insert with array generation
        await db_conn.execute("""
            INSERT INTO array_test
            SELECT
                ARRAY[i::float4, (i+1)::float4, (i+2)::float4],
                ARRAY[[i::float4, (i+1)::float4], [(i+2)::float4, (i+3)::float4]],
                ARRAY[('\\x' || lpad(to_hex(i % 256), 2, '0'))::bytea,
                      ('\\x' || lpad(to_hex((i+1) % 256), 2, '0'))::bytea,
                      ('\\x' || lpad(to_hex((i+2) % 256), 2, '0'))::bytea],
                ARRAY['text' || i::text, 'text' || (i+1)::text, 'text' || (i+2)::text],
                ARRAY['varchar' || i::text, 'varchar' || (i+1)::text, 'varchar' || (i+2)::text]
            FROM generate_series(1, 1000) i
        """)

        await assertions.assert_table_row_count(1007, "array_test")

        # Test filtering on bulk inserted data
        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM array_test WHERE float4_array_1d[1] = 100.0"
        )
        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM array_test WHERE float4_array_1d[2] = 501.0"
        )
        await assertions.assert_query_row_count(
            5,
            "SELECT * FROM array_test WHERE 10.0 = ANY(float4_array_1d)"
        )

        print("✓ Test passed: All array operations work correctly")

    finally:
        # Cleanup
        await db_conn.execute("RESET pg_deeplake.use_deeplake_executor")
        await db_conn.execute("DROP TABLE IF EXISTS array_test CASCADE")


@pytest.mark.asyncio
@pytest.mark.parametrize("data_type,test_data", [
    ("int4", [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]], [[13, 14, 15], [16, 17, 18]]]),
    ("float4", [[[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]], [[7.7, 8.8, 9.9], [10.1, 11.1, 12.1]], [[13.1, 14.1, 15.1], [16.1, 17.1, 18.1]]]),
])
async def test_2d_array_single_column(db_conn: asyncpg.Connection, data_type: str, test_data: list):
    """
    Test 2D array operations with a single column table.

    Tests:
    - Creating table with single 2D array column
    - Inserting 2D array data
    - Reading back and verifying data integrity
    - Parametrized for int4 and float4 types
    """
    assertions = Assertions(db_conn)
    table_name = f"array_2d_test_{data_type}"

    try:
        # Create table with ID column and single 2D array column
        await db_conn.execute(f"""
            CREATE TABLE {table_name} (
                id INT,
                array_2d {data_type}[][]
            ) USING deeplake
        """)

        # Insert test data with explicit IDs
        for idx, data in enumerate(test_data):
            await db_conn.execute(
                f"INSERT INTO {table_name} (id, array_2d) VALUES ($1, $2::{data_type}[][])",
                idx, data
            )

        # Verify row count
        await assertions.assert_table_row_count(len(test_data), table_name)

        # Read back and verify data, ordered by ID to match insertion order
        rows = await db_conn.fetch(f"SELECT id, array_2d FROM {table_name} ORDER BY id")

        assert len(rows) == len(test_data), f"Expected {len(test_data)} rows, got {len(rows)}"

        for idx, row in enumerate(rows):
            retrieved_data = row['array_2d']
            expected_data = test_data[idx]

            # Debug: print what we got
            print(f"Row {idx}: type={type(retrieved_data)}, value={retrieved_data}")

            # Convert to nested lists for comparison if needed
            if retrieved_data is not None:
                # Check if it's already a list structure
                if isinstance(retrieved_data, list):
                    retrieved_list = [list(inner) if hasattr(inner, '__iter__') and not isinstance(inner, str) else inner for inner in retrieved_data]
                else:
                    retrieved_list = retrieved_data

                print(f"Row {idx}: Expected {expected_data}, got {retrieved_list}")
                
                # Use approximate comparison for floats (float4 has ~7 significant digits)
                if data_type == "float4":
                    for i, (expected_row, retrieved_row) in enumerate(zip(expected_data, retrieved_list)):
                        for j, (expected_val, retrieved_val) in enumerate(zip(expected_row, retrieved_row)):
                            assert abs(expected_val - retrieved_val) < 1e-4, \
                                f"Row {idx}[{i}][{j}]: Expected {expected_val}, got {retrieved_val}"
                else:
                    assert retrieved_list == expected_data, \
                        f"Row {idx}: Expected {expected_data}, got {retrieved_list}"

        # Test element access
        if data_type == "int4":
            await assertions.assert_query_row_count(
                1,
                f"SELECT * FROM {table_name} WHERE array_2d[1][1] = 1"
            )
            await assertions.assert_query_row_count(
                1,
                f"SELECT * FROM {table_name} WHERE array_2d[2][3] = 6"
            )
        else:  # float4
            await assertions.assert_query_row_count(
                1,
                f"SELECT * FROM {table_name} WHERE array_2d[1][1] = 1.1"
            )
            await assertions.assert_query_row_count(
                1,
                f"SELECT * FROM {table_name} WHERE array_2d[2][3] = 6.6"
            )

        print(f"✓ Test passed: 2D {data_type} array operations work correctly")

    finally:
        # Cleanup
        await db_conn.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE")


@pytest.mark.asyncio
@pytest.mark.parametrize("data_type,test_data", [
    ("int4", [[1, 2, 3, 4, 5], [10, 20, 30, 40, 50], [100, 200, 300, 400, 500]]),
    ("float4", [[1.1, 2.2, 3.3, 4.4, 5.5], [10.1, 20.2, 30.3, 40.4, 50.5], [100.1, 200.2, 300.3, 400.4, 500.5]]),
])
async def test_1d_array_single_column(db_conn: asyncpg.Connection, data_type: str, test_data: list):
    """
    Test 1D array operations with a single column table.

    Tests:
    - Creating table with single 1D array column
    - Inserting 1D array data
    - Reading back and verifying data integrity
    - Parametrized for int4 and float4 types
    """
    assertions = Assertions(db_conn)
    table_name = f"array_1d_test_{data_type}"

    try:
        # Create table with ID column and single 1D array column
        await db_conn.execute(f"""
            CREATE TABLE {table_name} (
                id INT,
                array_1d {data_type}[]
            ) USING deeplake
        """)

        # Insert test data with explicit IDs
        for idx, data in enumerate(test_data):
            await db_conn.execute(
                f"INSERT INTO {table_name} (id, array_1d) VALUES ($1, $2::{data_type}[])",
                idx, data
            )

        # Verify row count
        await assertions.assert_table_row_count(len(test_data), table_name)

        # Read back and verify data, ordered by ID to match insertion order
        rows = await db_conn.fetch(f"SELECT id, array_1d FROM {table_name} ORDER BY id")

        assert len(rows) == len(test_data), f"Expected {len(test_data)} rows, got {len(rows)}"

        for idx, row in enumerate(rows):
            retrieved_data = row['array_1d']
            expected_data = test_data[idx]

            # Debug: print what we got
            print(f"Row {idx}: type={type(retrieved_data)}, value={retrieved_data}")

            # Convert to list for comparison if needed
            if retrieved_data is not None:
                retrieved_list = list(retrieved_data) if hasattr(retrieved_data, '__iter__') else retrieved_data

                print(f"Row {idx}: Expected {expected_data}, got {retrieved_list}")

                # Use approximate comparison for floats (float4 has ~7 significant digits)
                if data_type == "float4":
                    assert len(retrieved_list) == len(expected_data), \
                        f"Row {idx}: Length mismatch - Expected {len(expected_data)}, got {len(retrieved_list)}"
                    for i, (expected_val, retrieved_val) in enumerate(zip(expected_data, retrieved_list)):
                        assert abs(expected_val - retrieved_val) < 1e-4, \
                            f"Row {idx}[{i}]: Expected {expected_val}, got {retrieved_val}"
                else:
                    assert retrieved_list == expected_data, \
                        f"Row {idx}: Expected {expected_data}, got {retrieved_list}"

        # Test element access
        if data_type == "int4":
            await assertions.assert_query_row_count(
                1,
                f"SELECT * FROM {table_name} WHERE array_1d[1] = 1"
            )
            await assertions.assert_query_row_count(
                1,
                f"SELECT * FROM {table_name} WHERE array_1d[3] = 300"
            )
        else:  # float4
            await assertions.assert_query_row_count(
                1,
                f"SELECT * FROM {table_name} WHERE array_1d[1] = 1.1"
            )
            await assertions.assert_query_row_count(
                1,
                f"SELECT * FROM {table_name} WHERE array_1d[3] = 300.3"
            )

        print(f"✓ Test passed: 1D {data_type} array operations work correctly")

    finally:
        # Cleanup
        await db_conn.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE")
