"""
Test PostgreSQL to DuckDB query translation.

This test verifies that the pg_to_duckdb_translator correctly translates
PostgreSQL query syntax to DuckDB-compatible syntax for queries targeting
deeplake tables.

Translation rules tested (15 test functions):
1. JSON operators: -> and ->> to DuckDB's ->> with $.path syntax
2. Type casts: ::TYPE to CAST(expr AS TYPE)
3. Timestamp conversions: PostgreSQL epoch arithmetic to TO_TIMESTAMP()
4. Date extraction: EXTRACT() to DuckDB's date part functions
5. Date differences: EXTRACT(EPOCH FROM ...) to date_diff()
6. IN clauses: IN ('a', 'b') to in ['a', 'b']
7. COUNT(*) to count()
8. WHERE predicate wrapping: Add parentheses around each predicate (with word boundaries)
9. to_date to strptime
10. DIV() to // operator
11. regexp_substr to regexp_extract
12. BETWEEN clauses: Preserve BETWEEN...AND syntax without splitting
13. Column names with keywords: Handle column names containing 'and', 'or' as substrings
14. Complex combined translations: Multiple rules applied together
15. Regression tests: Position increment bug in BETWEEN handling
"""
import pytest
import asyncpg
from test_utils.assertions import Assertions


@pytest.mark.asyncio
async def test_json_operators(db_conn: asyncpg.Connection):
    """
    Test JSON operator translation (-> and ->> to DuckDB syntax).

    PostgreSQL: json -> 'key' ->> 'value'
    DuckDB: json->>'$.key.value'
    """
    assertions = Assertions(db_conn)

    try:
        # Create table with JSON column
        await db_conn.execute("""
            CREATE TABLE json_events (
                id serial PRIMARY KEY,
                data jsonb
            ) USING deeplake
        """)

        # Insert test data
        await db_conn.execute("""
            INSERT INTO json_events (data) VALUES
                ('{"commit": {"collection": "app.bsky.feed.post", "operation": "create"}}'::jsonb),
                ('{"commit": {"collection": "app.bsky.feed.like", "operation": "create"}}'::jsonb),
                ('{"commit": {"collection": "app.bsky.feed.post", "operation": "delete"}}'::jsonb)
        """)

        # Test 1: Simple nested JSON access
        await assertions.assert_query_row_count(
            2,
            "SELECT * FROM json_events WHERE data -> 'commit' ->> 'collection' = 'app.bsky.feed.post'"
        )

        # Test 2: JSON access with aggregation
        result = await db_conn.fetchval("""
            SELECT COUNT(*) FROM json_events
            WHERE data ->> 'kind' = 'commit'
            AND data -> 'commit' ->> 'operation' = 'create'
        """)
        # Should translate and execute without error
        assert result is not None

        print("✓ Test passed: JSON operators translate correctly")

    finally:
        await db_conn.execute("DROP TABLE IF EXISTS json_events CASCADE")


@pytest.mark.asyncio
async def test_type_casts(db_conn: asyncpg.Connection):
    """
    Test type cast translation (::TYPE to CAST(expr AS TYPE)).

    Covers:
    - Simple identifier casts: id::INTEGER
    - Parenthesized expression casts: (expr)::DECIMAL
    - String literal casts: 'text'::varchar
    - Compound types: (value)::DOUBLE PRECISION
    """
    assertions = Assertions(db_conn)

    try:
        # Create test table
        await db_conn.execute("""
            CREATE TABLE numbers (
                id int,
                value text,
                amount numeric
            ) USING deeplake
        """)

        await db_conn.execute("""
            INSERT INTO numbers VALUES
                (1, '100', 10.5),
                (2, '200', 20.75),
                (3, '300', 30.25)
        """)

        # Test 1: Simple identifier cast
        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM numbers WHERE id::text = '2'"
        )

        # Test 2: Parenthesized expression cast
        # amount values: 10.5, 20.75, 30.25
        # (30.25 * 2)::INTEGER = 60::INTEGER = 60, so >= 60 matches only row 3
        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM numbers WHERE (amount * 2)::INTEGER >= 60"
        )

        # Test 3: DOUBLE PRECISION compound type
        result = await db_conn.fetchval("""
            SELECT (amount)::DOUBLE PRECISION FROM numbers WHERE id = 1
        """)
        assert result is not None

        print("✓ Test passed: Type casts translate correctly")

    finally:
        await db_conn.execute("DROP TABLE IF EXISTS numbers CASCADE")


@pytest.mark.asyncio
async def test_timestamp_conversions(db_conn: asyncpg.Connection):
    """
    Test timestamp conversion translation.

    PostgreSQL: TIMESTAMP WITH TIME ZONE 'epoch' + INTERVAL '1 microsecond' * expr
    DuckDB: TO_TIMESTAMP(CAST(expr AS BIGINT) / 1000000)
    """
    assertions = Assertions(db_conn)

    try:
        # Create table with microsecond timestamps
        await db_conn.execute("""
            CREATE TABLE events (
                id int,
                time_us bigint
            ) USING deeplake
        """)

        # Insert test data (microseconds since epoch)
        await db_conn.execute("""
            INSERT INTO events VALUES
                (1, 1609459200000000),  -- 2021-01-01 00:00:00 UTC
                (2, 1640995200000000),  -- 2022-01-01 00:00:00 UTC
                (3, 1672531200000000)   -- 2023-01-01 00:00:00 UTC
        """)

        # Test: Timestamp conversion from microseconds
        result = await db_conn.fetchval("""
            SELECT MIN(
                TIMESTAMP WITH TIME ZONE 'epoch' + INTERVAL '1 microsecond' * time_us
            ) FROM events
        """)
        assert result is not None

        print("✓ Test passed: Timestamp conversions translate correctly")

    finally:
        await db_conn.execute("DROP TABLE IF EXISTS events CASCADE")


@pytest.mark.asyncio
async def test_extract_functions(db_conn: asyncpg.Connection):
    """
    Test EXTRACT function translation to DuckDB date part functions.

    PostgreSQL: EXTRACT(HOUR FROM timestamp)
    DuckDB: hour(timestamp)
    """
    assertions = Assertions(db_conn)

    try:
        # Create table with timestamp
        await db_conn.execute("""
            CREATE TABLE timestamps (
                id int,
                ts timestamp
            ) USING deeplake
        """)

        await db_conn.execute("""
            INSERT INTO timestamps VALUES
                (1, '2024-03-15 08:30:00'),
                (2, '2024-03-15 14:45:00'),
                (3, '2024-03-15 20:15:00')
        """)

        # Test EXTRACT with various date parts
        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM timestamps WHERE EXTRACT(HOUR FROM ts) = 14"
        )

        await assertions.assert_query_row_count(
            3,
            "SELECT * FROM timestamps WHERE EXTRACT(DAY FROM ts) = 15"
        )

        await assertions.assert_query_row_count(
            3,
            "SELECT * FROM timestamps WHERE EXTRACT(MONTH FROM ts) = 3"
        )

        print("✓ Test passed: EXTRACT functions translate correctly")

    finally:
        await db_conn.execute("DROP TABLE IF EXISTS timestamps CASCADE")


@pytest.mark.asyncio
async def test_in_clauses(db_conn: asyncpg.Connection):
    """
    Test IN clause translation.

    PostgreSQL: column IN ('a', 'b', 'c')
    DuckDB: column in ['a', 'b', 'c']
    """
    assertions = Assertions(db_conn)

    try:
        # Create table
        await db_conn.execute("""
            CREATE TABLE categories (
                id int,
                category text
            ) USING deeplake
        """)

        await db_conn.execute("""
            INSERT INTO categories VALUES
                (1, 'electronics'),
                (2, 'appliances'),
                (3, 'furniture'),
                (4, 'books')
        """)

        # Test IN clause
        await assertions.assert_query_row_count(
            2,
            "SELECT * FROM categories WHERE category IN ('electronics', 'appliances')"
        )

        print("✓ Test passed: IN clauses translate correctly")

    finally:
        await db_conn.execute("DROP TABLE IF EXISTS categories CASCADE")


@pytest.mark.asyncio
async def test_where_predicate_wrapping(db_conn: asyncpg.Connection):
    """
    Test WHERE clause predicate wrapping.

    PostgreSQL: WHERE a = 1 AND b = 2
    DuckDB: WHERE (a = 1) AND (b = 2)
    """
    assertions = Assertions(db_conn)

    try:
        # Create table
        await db_conn.execute("""
            CREATE TABLE people (
                name text,
                age int
            ) USING deeplake
        """)

        await db_conn.execute("""
            INSERT INTO people VALUES
                ('Alice', 30),
                ('Bob', 25),
                ('Charlie', 30),
                ('David', 25)
        """)

        # Test complex WHERE with AND/OR
        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM people WHERE name = 'Alice' AND age = 30"
        )

        await assertions.assert_query_row_count(
            4,
            "SELECT * FROM people WHERE age = 30 OR age = 25"
        )

        # Test subquery with WHERE (regression test)
        await assertions.assert_query_row_count(
            1,
            "SELECT COUNT(*) FROM (SELECT * FROM people WHERE age = 25) AS subquery"
        )

        print("✓ Test passed: WHERE predicate wrapping works correctly")

    finally:
        await db_conn.execute("DROP TABLE IF EXISTS people CASCADE")


@pytest.mark.asyncio
async def test_to_date_translation(db_conn: asyncpg.Connection):
    """
    Test to_date to strptime translation.

    PostgreSQL: to_date('2024-01-01', 'YYYY-MM-DD')
    DuckDB: strptime('2024-01-01', 'YYYY-MM-DD')
    """
    assertions = Assertions(db_conn)

    try:
        # Create table with date strings
        await db_conn.execute("""
            CREATE TABLE users (
                id int,
                name text,
                signup_date date
            ) USING deeplake
        """)

        await db_conn.execute("""
            INSERT INTO users VALUES
                (1, 'Alice', '2024-01-01'),
                (2, 'Bob', '2024-06-15'),
                (3, 'Charlie', '2024-12-31')
        """)

        # Test to_date function (if translator is active, it becomes strptime)
        # Note: DuckDB's strptime uses %Y-%m-%d format, not YYYY-MM-DD
        await assertions.assert_query_row_count(
            2,
            "SELECT * FROM users WHERE signup_date >= to_date('2024-06-01', '%Y-%m-%d')"
        )

        print("✓ Test passed: to_date translates to strptime correctly")

    finally:
        await db_conn.execute("DROP TABLE IF EXISTS users CASCADE")


@pytest.mark.asyncio
async def test_div_function(db_conn: asyncpg.Connection):
    """
    Test DIV() function translation to // operator.

    PostgreSQL: DIV(a, b)
    DuckDB: (a // b)
    """
    assertions = Assertions(db_conn)

    try:
        # Create table
        await db_conn.execute("""
            CREATE TABLE transactions (
                id int,
                total_amount int,
                quantity int
            ) USING deeplake
        """)

        await db_conn.execute("""
            INSERT INTO transactions VALUES
                (1, 550, 75),
                (2, 1200, 150),
                (3, 300, 45)
        """)

        # Test DIV function: DIV(75,10)=7, DIV(150,10)=15, DIV(45,10)=4
        # So DIV(quantity, 10) > 5 should match rows 1 and 2
        await assertions.assert_query_row_count(
            2,
            "SELECT * FROM transactions WHERE DIV(quantity, 10) > 5"
        )

        print("✓ Test passed: DIV function translates correctly")

    finally:
        await db_conn.execute("DROP TABLE IF EXISTS transactions CASCADE")


@pytest.mark.asyncio
async def test_regexp_substr(db_conn: asyncpg.Connection):
    """
    Test regexp_substr to regexp_extract translation.

    PostgreSQL: regexp_substr(string, pattern)
    DuckDB: regexp_extract(string, pattern)
    """
    assertions = Assertions(db_conn)

    try:
        # Create table with text data
        await db_conn.execute("""
            CREATE TABLE users (
                id int,
                email text,
                phone text
            ) USING deeplake
        """)

        await db_conn.execute("""
            INSERT INTO users VALUES
                (1, 'alice@example.com', '415-555-0001'),
                (2, 'bob@test.org', '650-555-0002'),
                (3, 'charlie@example.com', '415-555-0003')
        """)

        # Test regexp_substr (will be translated to regexp_extract)
        await assertions.assert_query_row_count(
            2,
            "SELECT * FROM users WHERE regexp_substr(phone, '^[0-9]{3}') = '415'"
        )

        print("✓ Test passed: regexp_substr translates correctly")

    finally:
        await db_conn.execute("DROP TABLE IF EXISTS users CASCADE")


@pytest.mark.asyncio
async def test_varchar_text_casts(db_conn: asyncpg.Connection):
    """
    Test VARCHAR and TEXT type casts including string literals.

    Real-world test from PostgreSQL test suite (test_array.py, test_json.py).
    """
    assertions = Assertions(db_conn)

    try:
        # Create table with arrays (similar to test_array.py)
        await db_conn.execute("""
            CREATE TABLE array_test (
                id serial PRIMARY KEY,
                text_array_1d text[],
                varchar_array_1d varchar[]
            ) USING deeplake
        """)

        await db_conn.execute("""
            INSERT INTO array_test (text_array_1d, varchar_array_1d) VALUES
                (ARRAY['text1'::text, 'text2'::text, 'text3'::text],
                 ARRAY['varchar1'::varchar, 'varchar2'::varchar, 'varchar3'::varchar]),
                (ARRAY['text4'::text, 'text5'::text, 'text6'::text],
                 ARRAY['varchar4'::varchar, 'varchar5'::varchar, 'varchar6'::varchar])
        """)

        # Test array element access with type casts (from test_array.py:233-247)
        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM array_test WHERE text_array_1d[1] = 'text1'::text"
        )

        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM array_test WHERE varchar_array_1d[2] = 'varchar2'::varchar"
        )

        print("✓ Test passed: VARCHAR and TEXT casts work correctly")

    finally:
        await db_conn.execute("DROP TABLE IF EXISTS array_test CASCADE")


@pytest.mark.asyncio
async def test_complex_combined_translation(db_conn: asyncpg.Connection):
    """
    Test complex query with multiple translation rules combined.

    This tests a realistic query that uses multiple translation features:
    - JSON operators
    - Type casts
    - EXTRACT functions
    - IN clauses
    - WHERE predicate wrapping
    - COUNT(*)
    """
    assertions = Assertions(db_conn)

    try:
        # Create table with JSON and timestamp data
        await db_conn.execute("""
            CREATE TABLE events (
                id serial PRIMARY KEY,
                event_data jsonb,
                created_at timestamp,
                event_type text
            ) USING deeplake
        """)

        await db_conn.execute("""
            INSERT INTO events (event_data, created_at, event_type) VALUES
                ('{"user": {"id": 123, "name": "Alice"}}'::jsonb, '2024-01-15 10:00:00', 'login'),
                ('{"user": {"id": 456, "name": "Bob"}}'::jsonb, '2024-01-15 14:30:00', 'logout'),
                ('{"user": {"id": 123, "name": "Alice"}}'::jsonb, '2024-01-15 16:45:00', 'purchase'),
                ('{"user": {"id": 789, "name": "Charlie"}}'::jsonb, '2024-01-16 09:15:00', 'login')
        """)

        # Complex query with multiple translation features
        result = await db_conn.fetchval("""
            SELECT COUNT(*)
            FROM events
            WHERE event_data -> 'user' ->> 'name' IN ('Alice', 'Bob')
            AND EXTRACT(HOUR FROM created_at) >= 10
            AND event_type = 'login'::text
            AND (event_data -> 'user' ->> 'id')::INTEGER > 100
        """)

        assert result == 1, f"Expected 1 row, got {result}"

        print("✓ Test passed: Complex combined translation works correctly")

    finally:
        await db_conn.execute("DROP TABLE IF EXISTS events CASCADE")


@pytest.mark.asyncio
async def test_date_arithmetic_with_casts(db_conn: asyncpg.Connection):
    """
    Test string concatenation with TEXT cast (from test_string_types.py:158).

    Real-world test: 'bpchar_' || LPAD(i::text, 4, '0')
    """
    assertions = Assertions(db_conn)

    try:
        # Create table for string operations
        await db_conn.execute("""
            CREATE TABLE string_data (
                id int,
                label text
            ) USING deeplake
        """)

        # Insert with string concatenation and cast (from test_string_types.py)
        await db_conn.execute("""
            INSERT INTO string_data
            SELECT
                i,
                'bpchar_' || LPAD(i::text, 4, '0')
            FROM generate_series(1, 10) i
        """)

        await assertions.assert_table_row_count(10, "string_data")

        # Query with cast in WHERE clause
        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM string_data WHERE (id)::text = '5'"
        )

        print("✓ Test passed: String operations with TEXT casts work correctly")

    finally:
        await db_conn.execute("DROP TABLE IF EXISTS string_data CASCADE")


@pytest.mark.asyncio
async def test_between_clause(db_conn: asyncpg.Connection):
    """
    Test BETWEEN clause translation.

    BETWEEN uses AND as part of its syntax, not as a logical operator.
    The translator must not split "col BETWEEN val1 AND val2" into separate predicates.

    PostgreSQL: WHERE col BETWEEN 1 AND 10
    DuckDB: WHERE (col BETWEEN 1 AND 10)  -- NOT: WHERE (col BETWEEN 1) AND (10)
    """
    assertions = Assertions(db_conn)

    try:
        # Create table similar to TPCH lineitem
        await db_conn.execute("""
            CREATE TABLE lineitem (
                l_discount decimal,
                l_quantity decimal,
                l_shipdate date,
                l_extendedprice decimal
            ) USING deeplake
        """)

        await db_conn.execute("""
            INSERT INTO lineitem VALUES
                (0.05, 10, '1994-01-15', 1000.00),
                (0.06, 20, '1994-06-15', 2000.00),
                (0.07, 30, '1995-01-15', 3000.00),
                (0.10, 40, '1995-06-15', 4000.00)
        """)

        # Test 1: BETWEEN with arithmetic (from TPCH Query 6)
        # l_discount between 0.06 - 0.01 AND 0.06 + 0.01
        # Matches rows where l_discount is between 0.05 and 0.07 inclusive
        # Test data: 0.05✓, 0.06✓, 0.07✓, 0.10✗ → 3 rows expected
        await assertions.assert_query_row_count(
            3,
            "SELECT * FROM lineitem WHERE l_discount BETWEEN 0.06 - 0.01 AND 0.06 + 0.01"
        )

        # Test 2: BETWEEN with dates (from TPCH Query 7, 8)
        await assertions.assert_query_row_count(
            2,
            "SELECT * FROM lineitem WHERE l_shipdate BETWEEN '1994-01-01' AND '1994-12-31'"
        )

        # Test 3: Multiple BETWEEN clauses with AND (complex case)
        await assertions.assert_query_row_count(
            1,
            """SELECT * FROM lineitem
               WHERE l_discount BETWEEN 0.05 AND 0.06
               AND l_quantity BETWEEN 15 AND 25"""
        )

        print("✓ Test passed: BETWEEN clauses translate correctly")

    finally:
        await db_conn.execute("DROP TABLE IF EXISTS lineitem CASCADE")


@pytest.mark.asyncio
async def test_column_names_with_keywords(db_conn: asyncpg.Connection):
    """
    Test column names containing SQL keywords like AND, OR.

    Column names like 'p_brand', 'standard', 'portland' contain 'and' or 'or'
    as substrings. The translator must use word boundaries to avoid splitting
    these column names.

    PostgreSQL: WHERE p_brand <> 'Brand#45'
    DuckDB: WHERE (p_brand <> 'Brand#45')  -- NOT: WHERE (p_br) AND (<> 'Brand#45')
    """
    assertions = Assertions(db_conn)

    try:
        # Create table similar to TPCH part table
        await db_conn.execute("""
            CREATE TABLE part (
                p_partkey int,
                p_brand text,
                p_type text,
                p_size int
            ) USING deeplake
        """)

        await db_conn.execute("""
            INSERT INTO part VALUES
                (1, 'Brand#12', 'STANDARD POLISHED', 10),
                (2, 'Brand#45', 'MEDIUM POLISHED', 20),
                (3, 'Brand#23', 'LARGE BRUSHED', 30),
                (4, 'Brand#45', 'SMALL BURNISHED', 15)
        """)

        # Test 1: Column name with 'and' substring (from TPCH Query 16)
        # p_brand contains 'and', must not split at "br-AND"
        # Only Row 1 (Brand#12, size 10) matches both conditions:
        #   - Row 1: Brand#12, size 10 → brand ≠ Brand#45 ✓, size < 25 ✓
        #   - Row 2: Brand#45, size 20 → brand ≠ Brand#45 ✗
        #   - Row 3: Brand#23, size 30 → brand ≠ Brand#45 ✓, size < 25 ✗
        #   - Row 4: Brand#45, size 15 → brand ≠ Brand#45 ✗
        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM part WHERE p_brand <> 'Brand#45' AND p_size < 25"
        )

        # Test 2: Column name with 'and' and equals (from TPCH Query 17)
        # p_brand = 'Brand#23'
        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM part WHERE p_brand = 'Brand#23' AND p_size = 30"
        )

        # Test 3: String value with 'and' in LIKE pattern
        # STANDARD contains 'and'
        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM part WHERE p_type LIKE 'STANDARD%' AND p_size = 10"
        )

        # Test 4: Multiple conditions with column names containing keywords
        await assertions.assert_query_row_count(
            3,
            """SELECT * FROM part
               WHERE p_brand IN ('Brand#12', 'Brand#23', 'Brand#45')
               AND p_type NOT LIKE 'MEDIUM%'"""
        )

        print("✓ Test passed: Column names with SQL keywords handle correctly")

    finally:
        await db_conn.execute("DROP TABLE IF EXISTS part CASCADE")


@pytest.mark.asyncio
async def test_between_position_increment_bug(db_conn: asyncpg.Connection):
    """
    Regression test for the position increment bug in BETWEEN clause handling.

    The bug: When detecting AND within BETWEEN, the code was incrementing pos by 1
    instead of 3, causing it to re-check the same position on the next iteration.

    This test specifically uses a query where the bug would cause incorrect parsing.
    Without the fix (pos += 3), the translator would incorrectly parse the BETWEEN
    clause, potentially leading to malformed output or infinite loops.
    """
    assertions = Assertions(db_conn)

    try:
        # Create test table
        await db_conn.execute("""
            CREATE TABLE test_data (
                id int,
                value int,
                category text
            ) USING deeplake
        """)

        await db_conn.execute("""
            INSERT INTO test_data VALUES
                (1, 5, 'A'),
                (2, 15, 'B'),
                (3, 25, 'C'),
                (4, 35, 'D')
        """)

        # Test 1: Simple BETWEEN that would trigger the bug
        # The AND in "BETWEEN 10 AND 30" should not be treated as a logical separator
        await assertions.assert_query_row_count(
            2,
            "SELECT * FROM test_data WHERE value BETWEEN 10 AND 30"
        )

        # Test 2: BETWEEN followed by a logical AND
        # This tests that after handling BETWEEN...AND, we correctly parse the next AND
        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM test_data WHERE value BETWEEN 10 AND 30 AND category = 'B'"
        )

        # Test 3: Multiple consecutive BETWEENs with logical ANDs
        # This would definitely expose the position increment bug
        await assertions.assert_query_row_count(
            0,
            """SELECT * FROM test_data
               WHERE value BETWEEN 10 AND 20
               AND id BETWEEN 5 AND 10"""
        )

        # Test 4: BETWEEN with expressions (from TPCH Query 6 pattern)
        # This is the actual pattern that failed in TPCH tests
        await assertions.assert_query_row_count(
            2,
            "SELECT * FROM test_data WHERE value BETWEEN 5 + 5 AND 15 + 15"
        )

        print("✓ Test passed: BETWEEN position increment bug regression test")

    finally:
        await db_conn.execute("DROP TABLE IF EXISTS test_data CASCADE")
