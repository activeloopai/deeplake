"""
Test JSON and JSONB operations with deeplake storage.

Ported from: postgres/tests/sql/json.sql
"""
import pytest
import asyncpg
from test_utils.assertions import Assertions


@pytest.mark.asyncio
async def test_json_operations(db_conn: asyncpg.Connection):
    """
    Test JSON and JSONB data type operations with deeplake storage.

    Tests:
    - JSON and JSONB column storage
    - Exact matches for JSON/JSONB values
    - Field extraction using -> and ->>
    - Numeric comparisons on JSON fields
    - JSONB containment operator (@>)
    - JSONB existence operator (?)
    - Nested field extraction
    - Array containment in JSONB
    - Bulk inserts with JSON generation
    """
    assertions = Assertions(db_conn)

    try:
        # Create table with JSON columns
        await db_conn.execute("""
            CREATE TABLE json_test (
                json_col json,
                jsonb_col jsonb
            ) USING deeplake
        """)

        # Insert test data
        await db_conn.execute("""
            INSERT INTO json_test (json_col, jsonb_col)
            VALUES
                ('{"name": "Alice", "age": 30, "city": "NYC"}',
                 '{"name": "Alice", "age": 30, "city": "NYC"}'),
                ('{"name": "Bob", "age": 25, "city": "LA"}',
                 '{"name": "Bob", "age": 25, "city": "LA"}'),
                ('{"name": "Charlie", "age": 35, "city": "SF"}',
                 '{"name": "Charlie", "age": 35, "city": "SF"}'),
                ('{"name": "David", "age": 40, "city": "NYC"}',
                 '{"name": "David", "age": 40, "city": "NYC"}')
        """)

        await assertions.assert_table_row_count(4, "json_test")

        # Additional inserts including duplicates
        await db_conn.execute("""
            INSERT INTO json_test (json_col, jsonb_col)
            VALUES
                ('{"name": "David", "age": 40, "city": "NYC"}',
                 '{"name": "David", "age": 40, "city": "NYC"}'),
                ('{"name": "Eve", "age": 28, "city": "Boston"}',
                 '{"name": "Eve", "age": 28, "city": "Boston"}'),
                ('{"name": "Frank", "age": 32, "city": "Seattle"}',
                 '{"name": "Frank", "age": 32, "city": "Seattle"}')
        """)

        await assertions.assert_table_row_count(7, "json_test")

        # Test exact matches for JSON columns
        await assertions.assert_query_row_count(
            2,
            "SELECT * FROM json_test WHERE json_col::text = '{\"name\": \"David\", \"age\": 40, \"city\": \"NYC\"}'"
        )
        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM json_test WHERE json_col::text = '{\"name\": \"Alice\", \"age\": 30, \"city\": \"NYC\"}'"
        )

        # Test exact matches for JSONB columns
        await db_conn.execute("SET pg_deeplake.use_deeplake_executor = false")
        await assertions.assert_query_row_count(
            2,
            "SELECT * FROM json_test WHERE jsonb_col = '{\"name\": \"David\", \"age\": 40, \"city\": \"NYC\"}'::jsonb"
        )
        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM json_test WHERE jsonb_col = '{\"name\": \"Bob\", \"age\": 25, \"city\": \"LA\"}'::jsonb"
        )

        # Test JSON field extraction and filtering
        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM json_test WHERE json_col->>'name' = 'Alice'"
        )
        await assertions.assert_query_row_count(
            2,
            "SELECT * FROM json_test WHERE json_col->>'name' = 'David'"
        )
        await assertions.assert_query_row_count(
            3,
            "SELECT * FROM json_test WHERE json_col->>'city' = 'NYC'"
        )
        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM json_test WHERE json_col->>'city' = 'SF'"
        )

        # Test JSONB field extraction and filtering
        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM json_test WHERE jsonb_col->>'name' = 'Alice'"
        )
        await assertions.assert_query_row_count(
            2,
            "SELECT * FROM json_test WHERE jsonb_col->>'name' = 'David'"
        )
        await assertions.assert_query_row_count(
            3,
            "SELECT * FROM json_test WHERE jsonb_col->>'city' = 'NYC'"
        )
        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM json_test WHERE jsonb_col->>'city' = 'Boston'"
        )

        # Test numeric comparisons on JSON fields
        await assertions.assert_query_row_count(
            4,
            "SELECT * FROM json_test WHERE (json_col->>'age')::int > 30"
        )
        await assertions.assert_query_row_count(
            5,
            "SELECT * FROM json_test WHERE (json_col->>'age')::int >= 30"
        )
        await assertions.assert_query_row_count(
            2,
            "SELECT * FROM json_test WHERE (json_col->>'age')::int < 30"
        )
        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM json_test WHERE (json_col->>'age')::int = 25"
        )

        # Test numeric comparisons on JSONB fields
        await assertions.assert_query_row_count(
            4,
            "SELECT * FROM json_test WHERE (jsonb_col->>'age')::int > 30"
        )
        await assertions.assert_query_row_count(
            5,
            "SELECT * FROM json_test WHERE (jsonb_col->>'age')::int >= 30"
        )
        await assertions.assert_query_row_count(
            2,
            "SELECT * FROM json_test WHERE (jsonb_col->>'age')::int < 30"
        )
        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM json_test WHERE (jsonb_col->>'age')::int = 28"
        )

        # Test JSONB containment operator (@>)
        await assertions.assert_query_row_count(
            3,
            "SELECT * FROM json_test WHERE jsonb_col @> '{\"city\": \"NYC\"}'::jsonb"
        )
        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM json_test WHERE jsonb_col @> '{\"name\": \"Alice\", \"age\": 30}'::jsonb"
        )
        await assertions.assert_query_row_count(
            2,
            "SELECT * FROM json_test WHERE jsonb_col @> '{\"age\": 40}'::jsonb"
        )

        # Test JSONB existence operator (?)
        await assertions.assert_query_row_count(
            7,
            "SELECT * FROM json_test WHERE jsonb_col ? 'name'"
        )
        await assertions.assert_query_row_count(
            7,
            "SELECT * FROM json_test WHERE jsonb_col ? 'age'"
        )
        await assertions.assert_query_row_count(
            7,
            "SELECT * FROM json_test WHERE jsonb_col ? 'city'"
        )
        await assertions.assert_query_row_count(
            0,
            "SELECT * FROM json_test WHERE jsonb_col ? 'email'"
        )

        # Insert complex nested JSON structures
        await db_conn.execute("""
            INSERT INTO json_test (json_col, jsonb_col)
            VALUES
                ('{"person": {"name": "Grace", "details": {"age": 45, "hobbies": ["reading", "hiking"]}}}',
                 '{"person": {"name": "Grace", "details": {"age": 45, "hobbies": ["reading", "hiking"]}}}'),
                ('{"person": {"name": "Henry", "details": {"age": 50, "hobbies": ["cooking", "gaming"]}}}',
                 '{"person": {"name": "Henry", "details": {"age": 50, "hobbies": ["cooking", "gaming"]}}}')
        """)

        await assertions.assert_table_row_count(9, "json_test")

        # Test nested field extraction
        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM json_test WHERE json_col->'person'->>'name' = 'Grace'"
        )
        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM json_test WHERE jsonb_col->'person'->>'name' = 'Henry'"
        )
        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM json_test WHERE (json_col->'person'->'details'->>'age')::int = 45"
        )
        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM json_test WHERE (jsonb_col->'person'->'details'->>'age')::int = 50"
        )

        # Test array containment in JSONB
        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM json_test WHERE jsonb_col->'person'->'details'->'hobbies' @> '[\"reading\"]'::jsonb"
        )
        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM json_test WHERE jsonb_col->'person'->'details'->'hobbies' @> '[\"gaming\"]'::jsonb"
        )

        # Bulk insert with JSON generation
        await db_conn.execute("""
            INSERT INTO json_test
            SELECT
                json_build_object(
                    'id', i,
                    'name', 'User_' || i,
                    'score', (i * 10) % 100,
                    'active', (i % 2 = 0)
                ),
                jsonb_build_object(
                    'id', i,
                    'name', 'User_' || i,
                    'score', (i * 10) % 100,
                    'active', (i % 2 = 0)
                )
            FROM generate_series(1, 10000) i
        """)

        await assertions.assert_table_row_count(10009, "json_test")

        # Test filtering on bulk inserted data
        await assertions.assert_query_row_count(
            5000,
            "SELECT * FROM json_test WHERE (jsonb_col->>'active')::boolean = true"
        )
        await assertions.assert_query_row_count(
            5000,
            "SELECT * FROM json_test WHERE (jsonb_col->>'active')::boolean = false"
        )
        await assertions.assert_query_row_count(
            1000,
            "SELECT * FROM json_test WHERE (jsonb_col->>'score')::int = 50"
        )

        await db_conn.execute("RESET pg_deeplake.use_deeplake_executor")

        # Test null JSON and JSONB values
        await db_conn.execute("""
            INSERT INTO json_test (json_col, jsonb_col) VALUES (NULL, NULL);
        """)

        await assertions.assert_table_row_count(10010, "json_test")
        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM json_test WHERE json_col IS NULL"
        )
        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM json_test WHERE jsonb_col IS NULL"
        )

        print("✓ Test passed: All JSON operations work correctly")

    finally:
        # Cleanup
        await db_conn.execute("RESET pg_deeplake.use_deeplake_executor")
        await db_conn.execute("DROP TABLE IF EXISTS json_test CASCADE")
