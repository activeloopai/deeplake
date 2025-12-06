"""
Test TPC-H benchmark queries.

Ported from: postgres/tests/sql/tpch.sql

This test suite:
- Loads TPC-H schema from sql/tpch/create_schema.sql
- Inserts TPC-H data from sql/tpch/insert.sql
- Creates indexes for query performance
- Runs all 22 TPC-H queries from sql/tpch/1.sql through sql/tpch/22.sql
- Each query is a separate test function for individual execution

Usage:
    # Run all TPCH queries
    pytest -v -m tpch

    # Run specific query
    pytest -v test_tpch.py::test_tpch_query_1
    pytest -v test_tpch.py::test_tpch_query_6

    # Run queries 1-5
    pytest -v test_tpch.py::test_tpch_query_1 test_tpch.py::test_tpch_query_2 test_tpch.py::test_tpch_query_3 test_tpch.py::test_tpch_query_4 test_tpch.py::test_tpch_query_5
"""
import pytest
import asyncpg
import asyncio
from pathlib import Path
from typing import AsyncGenerator


def read_sql_file(file_path: Path, tests_dir: Path) -> str:
    """
    Read a SQL file and process \i (include) directives.

    Args:
        file_path: Path to the SQL file
        tests_dir: Tests directory for resolving relative includes

    Returns:
        SQL content with includes resolved
    """
    if not file_path.exists():
        raise FileNotFoundError(f"SQL file not found: {file_path}")

    with open(file_path, 'r') as f:
        lines = f.readlines()

    result = []
    for line in lines:
        stripped = line.strip()

        # Skip psql meta-commands except \i (include)
        if stripped.startswith('\\'):
            if stripped.startswith('\\i '):
                # Extract the include file path
                include_path = stripped[3:].strip()
                # Resolve relative to tests_dir (paths are like "sql/tpch/customer.sql")
                include_file = tests_dir / include_path
                # Recursively read the included file
                included_content = read_sql_file(include_file, tests_dir)
                result.append(included_content)
            # Skip other psql commands like \echo, \gset, \if, \endif, \quit, etc.
            continue

        result.append(line)

    return ''.join(result)


async def execute_sql_file(conn: asyncpg.Connection, sql_content: str) -> None:
    """
    Execute SQL content that may contain multiple statements.

    This function splits the SQL by semicolons and executes each statement
    individually, which is more reliable than passing everything to execute()
    at once. Handles DO blocks ($$..$$) and other multi-line constructs.

    Args:
        conn: Database connection
        sql_content: SQL content to execute
    """
    statements = []
    current_statement = []
    in_dollar_quote = False  # Track if we're inside a $$ ... $$ block
    dollar_quote_tag = None  # Store the dollar quote tag (e.g., $$ or $BODY$)

    for line in sql_content.split('\n'):
        stripped = line.strip()

        # Skip empty lines and comments (unless we're in a dollar quote block)
        if not in_dollar_quote and (not stripped or stripped.startswith('--')):
            continue

        current_statement.append(line)

        # Check for dollar quotes ($$, $tag$, etc.)
        # These are used in DO blocks, functions, etc. and can contain semicolons
        for i, char in enumerate(stripped):
            if char == '$':
                # Try to match a dollar quote tag
                end_idx = stripped.find('$', i + 1)
                if end_idx != -1:
                    tag = stripped[i:end_idx + 1]
                    if in_dollar_quote:
                        if tag == dollar_quote_tag:
                            in_dollar_quote = False
                            dollar_quote_tag = None
                    else:
                        in_dollar_quote = True
                        dollar_quote_tag = tag
                    break

        # Only treat semicolon as statement terminator if not in dollar quote
        if not in_dollar_quote and stripped.endswith(';'):
            stmt = '\n'.join(current_statement).strip()
            if stmt and not stmt.startswith('--'):
                statements.append(stmt)
            current_statement = []

    # Execute each statement
    for stmt in statements:
        if stmt.strip():
            try:
                await conn.execute(stmt)
            except Exception as e:
                # Print first 200 chars of failing statement for debugging
                print(f"Failed to execute statement: {stmt[:200]}...")
                raise


@pytest.fixture(scope="module")
def event_loop():
    """
    Create an event loop for the entire test module.

    This is required because tpch_db is module-scoped, but the default
    event_loop fixture from pytest-asyncio is function-scoped.
    """
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
async def tpch_db(pg_server) -> AsyncGenerator[asyncpg.Connection, None]:
    """
    Set up TPC-H database with schema, data, and indexes.

    This fixture runs once per test module and is shared by all TPC-H query tests.
    It:
    1. Creates a dedicated connection
    2. Drops and recreates pg_deeplake extension
    3. Creates TPC-H schema (8 tables + 1 view)
    4. Loads TPC-H data
    5. Creates indexes
    6. Yields the connection to all tests
    7. Cleans up on teardown
    """
    import os

    user = os.environ.get("USER", "postgres")
    conn = await asyncpg.connect(
        database="postgres",
        user=user,
        host="localhost"
    )

    # Get paths
    script_path = Path(__file__).parent  # py_tests/
    tests_dir = script_path.parent  # tests/
    tpch_dir = tests_dir / "sql" / "tpch"

    try:
        print("\n" + "=" * 70)
        print("TPC-H Database Setup (runs once for all queries)")
        print("=" * 70)

        # Clean extension state
        await conn.execute("DROP EXTENSION IF EXISTS pg_deeplake CASCADE")
        await conn.execute("CREATE EXTENSION pg_deeplake")

        # Step 1: Create schema
        print("\nStep 1: Creating TPC-H schema...")
        schema_sql = read_sql_file(tpch_dir / "create_schema.sql", tests_dir)
        await execute_sql_file(conn, schema_sql)
        print("✓ Schema created (8 tables + 1 view)")

        # Step 2: Insert data
        print("\nStep 2: Loading TPC-H data...")
        print("  This may take several minutes for large datasets...")
        insert_sql = read_sql_file(tpch_dir / "insert.sql", tests_dir)
        await execute_sql_file(conn, insert_sql)
        print("✓ Data loaded")

        # Step 3: Create indexes
        print("\nStep 3: Creating indexes...")
        await conn.execute("""
            CREATE UNIQUE INDEX idx_region_pk    ON region(r_regionkey);
            CREATE UNIQUE INDEX idx_nation_pk    ON nation(n_nationkey);
            CREATE UNIQUE INDEX idx_supplier_pk  ON supplier(s_suppkey);
            CREATE UNIQUE INDEX idx_customer_pk  ON customer(c_custkey);
            CREATE UNIQUE INDEX idx_orders_pk    ON orders(o_orderkey);
            CREATE UNIQUE INDEX idx_part_pk      ON part(p_partkey);
            CREATE UNIQUE INDEX idx_partsupp_pk  ON partsupp(ps_partkey, ps_suppkey);
            CREATE UNIQUE INDEX idx_lineitem_pk  ON lineitem(l_orderkey, l_linenumber);
            CREATE INDEX idx_deeplake_part_brand_container ON part USING btree (p_brand, p_container);
            CREATE INDEX idx_deeplake_lineitem_partkey_qty ON lineitem USING btree (l_partkey, l_quantity);
        """)
        print("✓ Indexes created")

        print("\n" + "=" * 70)
        print("✓ TPC-H database ready for queries")
        print("=" * 70 + "\n")

        yield conn

    finally:
        # Cleanup: Drop tables and view
        print("\n" + "=" * 70)
        print("TPC-H Database Cleanup")
        print("=" * 70)
        await conn.execute("DROP VIEW IF EXISTS revenue0 CASCADE")
        await conn.execute("DROP TABLE IF EXISTS customer CASCADE")
        await conn.execute("DROP TABLE IF EXISTS lineitem CASCADE")
        await conn.execute("DROP TABLE IF EXISTS nation CASCADE")
        await conn.execute("DROP TABLE IF EXISTS orders CASCADE")
        await conn.execute("DROP TABLE IF EXISTS part CASCADE")
        await conn.execute("DROP TABLE IF EXISTS partsupp CASCADE")
        await conn.execute("DROP TABLE IF EXISTS region CASCADE")
        await conn.execute("DROP TABLE IF EXISTS supplier CASCADE")
        await conn.execute("RESET pg_deeplake.use_deeplake_executor")
        await conn.close()
        print("✓ Cleanup complete\n")


def run_tpch_query(query_num: int, expected_count: int):
    """
    Helper to create a test function for a specific TPC-H query.

    Args:
        query_num: Query number (1-22)
        expected_count: Expected number of rows in result
    """
    async def test_func(tpch_db: asyncpg.Connection):
        # Get paths
        script_path = Path(__file__).parent  # py_tests/
        tests_dir = script_path.parent  # tests/
        tpch_dir = tests_dir / "sql" / "tpch"

        print(f"\nTPC-H Query {query_num}:")

        # Read query SQL
        query_file = tpch_dir / f"{query_num}.sql"
        query_sql = read_sql_file(query_file, tests_dir)

        # Execute query
        results = await tpch_db.fetch(query_sql)
        row_count = len(results)

        # Verify row count
        if row_count == expected_count:
            print(f"  ✓ PASSED: {row_count} rows (expected {expected_count})")
        else:
            raise AssertionError(
                f"Query {query_num} FAILED: Expected {expected_count} rows, got {row_count}"
            )

    # Set function metadata
    test_func.__name__ = f"test_tpch_query_{query_num}"
    test_func.__doc__ = f"""
    TPC-H Query {query_num}

    Expected result: {expected_count} rows

    Source: sql/tpch/{query_num}.sql
    """

    return test_func


# Generate individual test functions for each TPC-H query
# These can be run separately with: pytest test_tpch.py::test_tpch_query_N

@pytest.mark.asyncio
@pytest.mark.tpch
@pytest.mark.slow
async def test_tpch_query_1(tpch_db: asyncpg.Connection):
    """TPC-H Query 1: Pricing Summary Report (Expected: 4 rows)"""
    await run_tpch_query(1, 4)(tpch_db)


@pytest.mark.asyncio
@pytest.mark.tpch
@pytest.mark.slow
async def test_tpch_query_2(tpch_db: asyncpg.Connection):
    """TPC-H Query 2: Minimum Cost Supplier (Expected: 10 rows)"""
    await run_tpch_query(2, 10)(tpch_db)


@pytest.mark.asyncio
@pytest.mark.tpch
@pytest.mark.slow
async def test_tpch_query_3(tpch_db: asyncpg.Connection):
    """TPC-H Query 3: Shipping Priority (Expected: 10 rows)"""
    await run_tpch_query(3, 10)(tpch_db)


@pytest.mark.asyncio
@pytest.mark.tpch
@pytest.mark.slow
async def test_tpch_query_4(tpch_db: asyncpg.Connection):
    """TPC-H Query 4: Order Priority Checking (Expected: 5 rows)"""
    await run_tpch_query(4, 5)(tpch_db)


@pytest.mark.asyncio
@pytest.mark.tpch
@pytest.mark.slow
async def test_tpch_query_5(tpch_db: asyncpg.Connection):
    """TPC-H Query 5: Local Supplier Volume (Expected: 5 rows)"""
    await run_tpch_query(5, 5)(tpch_db)


@pytest.mark.asyncio
@pytest.mark.tpch
@pytest.mark.slow
async def test_tpch_query_6(tpch_db: asyncpg.Connection):
    """TPC-H Query 6: Forecasting Revenue Change (Expected: 1 row)"""
    await run_tpch_query(6, 1)(tpch_db)


@pytest.mark.asyncio
@pytest.mark.tpch
@pytest.mark.slow
async def test_tpch_query_7(tpch_db: asyncpg.Connection):
    """TPC-H Query 7: Volume Shipping (Expected: 4 rows)"""
    await run_tpch_query(7, 4)(tpch_db)


@pytest.mark.asyncio
@pytest.mark.tpch
@pytest.mark.slow
async def test_tpch_query_8(tpch_db: asyncpg.Connection):
    """TPC-H Query 8: National Market Share (Expected: 1 row)"""
    await run_tpch_query(8, 1)(tpch_db)


@pytest.mark.asyncio
@pytest.mark.tpch
@pytest.mark.slow
async def test_tpch_query_9(tpch_db: asyncpg.Connection):
    """TPC-H Query 9: Product Type Profit Measure (Expected: 151 rows)"""
    await run_tpch_query(9, 151)(tpch_db)


@pytest.mark.asyncio
@pytest.mark.tpch
@pytest.mark.slow
async def test_tpch_query_10(tpch_db: asyncpg.Connection):
    """TPC-H Query 10: Returned Item Reporting (Expected: 20 rows)"""
    await run_tpch_query(10, 20)(tpch_db)


@pytest.mark.asyncio
@pytest.mark.tpch
@pytest.mark.slow
async def test_tpch_query_11(tpch_db: asyncpg.Connection):
    """TPC-H Query 11: Important Stock Identification (Expected: 3127 rows)"""
    await run_tpch_query(11, 3127)(tpch_db)


@pytest.mark.asyncio
@pytest.mark.tpch
@pytest.mark.slow
async def test_tpch_query_12(tpch_db: asyncpg.Connection):
    """TPC-H Query 12: Shipping Modes and Order Priority (Expected: 2 rows)"""
    await run_tpch_query(12, 2)(tpch_db)


@pytest.mark.asyncio
@pytest.mark.tpch
@pytest.mark.slow
async def test_tpch_query_13(tpch_db: asyncpg.Connection):
    """TPC-H Query 13: Customer Distribution (Expected: 6 rows)"""
    await run_tpch_query(13, 6)(tpch_db)


@pytest.mark.asyncio
@pytest.mark.tpch
@pytest.mark.slow
async def test_tpch_query_14(tpch_db: asyncpg.Connection):
    """TPC-H Query 14: Promotion Effect (Expected: 1 row)"""
    await run_tpch_query(14, 1)(tpch_db)


@pytest.mark.asyncio
@pytest.mark.tpch
@pytest.mark.slow
async def test_tpch_query_15(tpch_db: asyncpg.Connection):
    """TPC-H Query 15: Top Supplier (Expected: 1 row)"""
    await run_tpch_query(15, 1)(tpch_db)


@pytest.mark.asyncio
@pytest.mark.tpch
@pytest.mark.slow
async def test_tpch_query_16(tpch_db: asyncpg.Connection):
    """TPC-H Query 16: Parts/Supplier Relationship (Expected: 10 rows)"""
    await run_tpch_query(16, 10)(tpch_db)


@pytest.mark.asyncio
@pytest.mark.tpch
@pytest.mark.slow
async def test_tpch_query_17(tpch_db: asyncpg.Connection):
    """TPC-H Query 17: Small-Quantity-Order Revenue (Expected: 1 row)"""
    await run_tpch_query(17, 1)(tpch_db)


@pytest.mark.asyncio
@pytest.mark.tpch
@pytest.mark.slow
async def test_tpch_query_18(tpch_db: asyncpg.Connection):
    """TPC-H Query 18: Large Volume Customer (Expected: 7 rows)"""
    await run_tpch_query(18, 7)(tpch_db)


@pytest.mark.asyncio
@pytest.mark.tpch
@pytest.mark.slow
async def test_tpch_query_19(tpch_db: asyncpg.Connection):
    """TPC-H Query 19: Discounted Revenue (Expected: 1 row)"""
    await run_tpch_query(19, 1)(tpch_db)


@pytest.mark.asyncio
@pytest.mark.tpch
@pytest.mark.slow
async def test_tpch_query_20(tpch_db: asyncpg.Connection):
    """TPC-H Query 20: Potential Part Promotion (Expected: 1 row)"""
    await run_tpch_query(20, 1)(tpch_db)


@pytest.mark.asyncio
@pytest.mark.tpch
@pytest.mark.slow
async def test_tpch_query_21(tpch_db: asyncpg.Connection):
    """TPC-H Query 21: Suppliers Who Kept Orders Waiting (Expected: 10 rows)"""
    await run_tpch_query(21, 10)(tpch_db)


@pytest.mark.asyncio
@pytest.mark.tpch
@pytest.mark.slow
async def test_tpch_query_22(tpch_db: asyncpg.Connection):
    """TPC-H Query 22: Global Sales Opportunity (Expected: 7 rows)"""
    await run_tpch_query(22, 7)(tpch_db)
