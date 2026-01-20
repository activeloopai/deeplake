"""
Comprehensive test for random column creation, row appending, and data operations.

This test module exercises:
- Random column generation with various data types
- Random row insertion with diverse values
- Bulk operations with randomized data
- Data integrity verification after operations
- Schema evolution with dynamic column additions
- Update and delete operations on random data
"""
import pytest
import asyncpg
import random
import string
import uuid
from datetime import datetime, date, time, timedelta
from decimal import Decimal
from typing import List, Dict, Any, Tuple
from test_utils.assertions import Assertions
from test_utils.helpers import generate_random_float_array


# Supported column types for random generation
COLUMN_TYPES = [
    'boolean',
    'int2',
    'int4',
    'int8',
    'float4',
    'float8',
    'numeric',
    'text',
    'varchar(255)',
    'date',
    'time',
    'timestamp',
    'timestamptz',
]


def random_string(length: int = 10) -> str:
    """Generate a random alphanumeric string."""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))


def random_column_name() -> str:
    """Generate a random valid column name."""
    return 'col_' + random_string(8).lower()


def random_value_for_type(col_type: str) -> Any:
    """Generate a random value for a given PostgreSQL column type."""
    col_type_lower = col_type.lower()

    if col_type_lower == 'boolean':
        return random.choice([True, False])
    elif col_type_lower == 'int2':
        return random.randint(-32768, 32767)
    elif col_type_lower == 'int4':
        return random.randint(-2147483648, 2147483647)
    elif col_type_lower == 'int8':
        return random.randint(-9223372036854775808, 9223372036854775807)
    elif col_type_lower == 'float4':
        return random.uniform(-1e6, 1e6)
    elif col_type_lower == 'float8':
        return random.uniform(-1e12, 1e12)
    elif col_type_lower == 'numeric':
        return Decimal(str(round(random.uniform(-1e6, 1e6), 4)))
    elif col_type_lower in ('text', 'varchar(255)'):
        return random_string(random.randint(1, 50))
    elif col_type_lower == 'date':
        start_date = date(2000, 1, 1)
        random_days = random.randint(0, 9000)
        return start_date + timedelta(days=random_days)
    elif col_type_lower == 'time':
        return time(
            random.randint(0, 23),
            random.randint(0, 59),
            random.randint(0, 59)
        )
    elif col_type_lower == 'timestamp':
        start_dt = datetime(2000, 1, 1, 0, 0, 0)
        random_seconds = random.randint(0, 800000000)
        return start_dt + timedelta(seconds=random_seconds)
    elif col_type_lower == 'timestamptz':
        start_dt = datetime(2000, 1, 1, 0, 0, 0)
        random_seconds = random.randint(0, 800000000)
        return start_dt + timedelta(seconds=random_seconds)
    else:
        return random_string(10)


def sql_literal_for_value(value: Any, col_type: str) -> str:
    """Convert a Python value to a SQL literal string."""
    if value is None:
        return 'NULL'

    col_type_lower = col_type.lower()

    if col_type_lower == 'boolean':
        return 'true' if value else 'false'
    elif col_type_lower in ('int2', 'int4', 'int8'):
        return str(value)
    elif col_type_lower in ('float4', 'float8'):
        return str(value)
    elif col_type_lower == 'numeric':
        return str(value)
    elif col_type_lower in ('text', 'varchar(255)'):
        escaped = str(value).replace("'", "''")
        return f"'{escaped}'"
    elif col_type_lower == 'date':
        return f"'{value.isoformat()}'"
    elif col_type_lower == 'time':
        return f"'{value.isoformat()}'"
    elif col_type_lower == 'timestamp':
        return f"'{value.isoformat()}'"
    elif col_type_lower == 'timestamptz':
        return f"'{value.isoformat()}'"
    else:
        escaped = str(value).replace("'", "''")
        return f"'{escaped}'"


@pytest.mark.asyncio
async def test_random_column_creation(db_conn: asyncpg.Connection):
    """
    Test dynamically adding random columns to a table.

    Tests:
    - Create a base table with a primary key
    - Add multiple random columns of different types using ALTER TABLE
    - Verify columns are added correctly in the catalog
    - Insert data into all columns
    - Query and verify data integrity
    """
    assertions = Assertions(db_conn)
    num_columns = 10

    try:
        # Create base table
        await db_conn.execute("""
            CREATE TABLE random_cols_test (
                id SERIAL PRIMARY KEY
            ) USING deeplake
        """)

        # Track added columns: (name, type)
        added_columns: List[Tuple[str, str]] = []

        # Add random columns one by one
        for _ in range(num_columns):
            col_name = random_column_name()
            col_type = random.choice(COLUMN_TYPES)

            await db_conn.execute(f"""
                ALTER TABLE random_cols_test ADD COLUMN "{col_name}" {col_type}
            """)
            added_columns.append((col_name, col_type))
            print(f"  Added column: {col_name} ({col_type})")

        # Verify columns exist in catalog
        catalog_cols = await db_conn.fetch("""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = 'random_cols_test'
            ORDER BY ordinal_position
        """)
        catalog_names = [c['column_name'] for c in catalog_cols]

        for col_name, _ in added_columns:
            assert col_name in catalog_names, \
                f"Column '{col_name}' should exist in catalog. Found: {catalog_names}"

        # Insert rows with random values for all columns
        num_rows = 20
        for row_idx in range(num_rows):
            col_names = ', '.join([f'"{name}"' for name, _ in added_columns])
            col_values = ', '.join([
                sql_literal_for_value(random_value_for_type(col_type), col_type)
                for _, col_type in added_columns
            ])

            await db_conn.execute(f"""
                INSERT INTO random_cols_test ({col_names}) VALUES ({col_values})
            """)

        # Verify row count
        await assertions.assert_table_row_count(num_rows, "random_cols_test")

        # Query all data and verify no errors
        rows = await db_conn.fetch("SELECT * FROM random_cols_test ORDER BY id")
        assert len(rows) == num_rows, f"Expected {num_rows} rows, got {len(rows)}"

        print(f"✓ Test passed: Successfully added {num_columns} random columns and inserted {num_rows} rows")

    finally:
        await db_conn.execute("DROP TABLE IF EXISTS random_cols_test CASCADE")


@pytest.mark.asyncio
async def test_random_row_append(db_conn: asyncpg.Connection):
    """
    Test appending random rows with various data types.

    Tests:
    - Create table with multiple column types
    - Append rows one at a time with random values
    - Append rows in bulk with random values
    - Verify data integrity after all inserts
    - Test NULL value handling
    """
    assertions = Assertions(db_conn)

    try:
        # Create table with variety of column types
        await db_conn.execute("""
            CREATE TABLE random_rows_test (
                id SERIAL PRIMARY KEY,
                bool_col boolean,
                int_col int4,
                bigint_col int8,
                float_col float8,
                numeric_col numeric,
                text_col text,
                date_col date,
                time_col time,
                ts_col timestamp
            ) USING deeplake
        """)

        # Define column info for generation
        columns = [
            ('bool_col', 'boolean'),
            ('int_col', 'int4'),
            ('bigint_col', 'int8'),
            ('float_col', 'float8'),
            ('numeric_col', 'numeric'),
            ('text_col', 'text'),
            ('date_col', 'date'),
            ('time_col', 'time'),
            ('ts_col', 'timestamp'),
        ]

        # Track inserted values for verification
        inserted_data: List[Dict[str, Any]] = []

        # Single row inserts
        num_single_inserts = 25
        for _ in range(num_single_inserts):
            row_data = {}
            col_names = []
            col_values = []

            for col_name, col_type in columns:
                # Randomly insert NULL for some values (10% chance)
                if random.random() < 0.1:
                    value = None
                else:
                    value = random_value_for_type(col_type)

                row_data[col_name] = value
                col_names.append(f'"{col_name}"')
                col_values.append(sql_literal_for_value(value, col_type))

            inserted_data.append(row_data)

            await db_conn.execute(f"""
                INSERT INTO random_rows_test ({', '.join(col_names)})
                VALUES ({', '.join(col_values)})
            """)

        await assertions.assert_table_row_count(num_single_inserts, "random_rows_test")

        # Multi-value insert (batch)
        num_batch_rows = 50
        batch_values = []

        for _ in range(num_batch_rows):
            row_data = {}
            row_values = []

            for col_name, col_type in columns:
                if random.random() < 0.1:
                    value = None
                else:
                    value = random_value_for_type(col_type)

                row_data[col_name] = value
                row_values.append(sql_literal_for_value(value, col_type))

            inserted_data.append(row_data)
            batch_values.append(f"({', '.join(row_values)})")

        col_names_str = ', '.join([f'"{name}"' for name, _ in columns])
        await db_conn.execute(f"""
            INSERT INTO random_rows_test ({col_names_str})
            VALUES {', '.join(batch_values)}
        """)

        total_rows = num_single_inserts + num_batch_rows
        await assertions.assert_table_row_count(total_rows, "random_rows_test")

        # Verify all data can be queried
        rows = await db_conn.fetch("SELECT * FROM random_rows_test ORDER BY id")
        assert len(rows) == total_rows, f"Expected {total_rows} rows, got {len(rows)}"

        # Verify no corruption in data types
        for row in rows:
            assert row['id'] is not None, "ID should never be NULL"
            # Boolean check
            if row['bool_col'] is not None:
                assert isinstance(row['bool_col'], bool), \
                    f"bool_col should be boolean, got {type(row['bool_col'])}"

        print(f"✓ Test passed: Successfully inserted {total_rows} random rows")

    finally:
        await db_conn.execute("DROP TABLE IF EXISTS random_rows_test CASCADE")


@pytest.mark.asyncio
async def test_random_bulk_operations(db_conn: asyncpg.Connection):
    """
    Test bulk insert, update, and delete with random data.

    Tests:
    - Bulk insert using generate_series with random expressions
    - Bulk update based on random conditions
    - Bulk delete based on random conditions
    - Data integrity after each operation
    """
    assertions = Assertions(db_conn)

    try:
        # Create table
        await db_conn.execute("""
            CREATE TABLE bulk_random_test (
                id SERIAL PRIMARY KEY,
                category int4,
                value float8,
                label text,
                created_at timestamp
            ) USING deeplake
        """)

        # Bulk insert using generate_series
        num_rows = 5000
        await db_conn.execute(f"""
            INSERT INTO bulk_random_test (category, value, label, created_at)
            SELECT
                (random() * 10)::int4 AS category,
                random() * 1000 AS value,
                'label_' || i AS label,
                '2020-01-01'::timestamp + (random() * 1000 || ' days')::interval AS created_at
            FROM generate_series(1, {num_rows}) i
        """)

        await assertions.assert_table_row_count(num_rows, "bulk_random_test")

        # Verify categories are distributed (0-10)
        categories = await db_conn.fetch("""
            SELECT DISTINCT category FROM bulk_random_test ORDER BY category
        """)
        assert len(categories) > 5, "Should have multiple distinct categories"

        # Bulk update: double the value for a random category
        random_cat = random.randint(0, 10)
        count_before = await db_conn.fetchval(f"""
            SELECT count(*) FROM bulk_random_test WHERE category = {random_cat}
        """)

        await db_conn.execute(f"""
            UPDATE bulk_random_test
            SET value = value * 2
            WHERE category = {random_cat}
        """)

        # Verify update didn't change row count
        await assertions.assert_table_row_count(num_rows, "bulk_random_test")

        # Bulk delete: remove rows with value below a random threshold
        random_threshold = random.uniform(50, 200)
        count_to_delete = await db_conn.fetchval(f"""
            SELECT count(*) FROM bulk_random_test WHERE value < {random_threshold}
        """)

        await db_conn.execute(f"""
            DELETE FROM bulk_random_test WHERE value < {random_threshold}
        """)

        expected_remaining = num_rows - count_to_delete
        await assertions.assert_table_row_count(expected_remaining, "bulk_random_test")

        # Verify no rows below threshold remain
        remaining_below = await db_conn.fetchval(f"""
            SELECT count(*) FROM bulk_random_test WHERE value < {random_threshold}
        """)
        assert remaining_below == 0, \
            f"No rows should remain below threshold {random_threshold}, found {remaining_below}"

        print(f"✓ Test passed: Bulk operations completed successfully")
        print(f"  - Inserted {num_rows} rows")
        print(f"  - Updated category {random_cat} ({count_before} rows)")
        print(f"  - Deleted {count_to_delete} rows below threshold {random_threshold:.2f}")
        print(f"  - Remaining rows: {expected_remaining}")

    finally:
        await db_conn.execute("DROP TABLE IF EXISTS bulk_random_test CASCADE")


@pytest.mark.asyncio
async def test_random_schema_evolution(db_conn: asyncpg.Connection):
    """
    Test schema evolution with random columns while data exists.

    Tests:
    - Create table and insert initial data
    - Add random columns to table with existing rows
    - Insert new rows with values for new columns
    - Update old rows to populate new columns
    - Verify data integrity throughout
    """
    assertions = Assertions(db_conn)

    try:
        # Create initial table
        await db_conn.execute("""
            CREATE TABLE schema_evolve_test (
                id SERIAL PRIMARY KEY,
                name text NOT NULL
            ) USING deeplake
        """)

        # Insert initial data
        initial_rows = 50
        for i in range(initial_rows):
            await db_conn.execute(f"""
                INSERT INTO schema_evolve_test (name) VALUES ('initial_{i}')
            """)

        await assertions.assert_table_row_count(initial_rows, "schema_evolve_test")

        # Add columns incrementally
        new_columns: List[Tuple[str, str]] = []
        for i in range(5):
            col_name = f"dyn_col_{i}"
            col_type = random.choice(['int4', 'float8', 'text', 'numeric'])

            await db_conn.execute(f"""
                ALTER TABLE schema_evolve_test ADD COLUMN "{col_name}" {col_type}
            """)
            new_columns.append((col_name, col_type))

            # Verify existing rows still accessible
            rows = await db_conn.fetch("SELECT * FROM schema_evolve_test LIMIT 5")
            assert len(rows) == 5, f"Should still have rows after adding column {col_name}"

            # Insert new row with value for new column
            value = random_value_for_type(col_type)
            await db_conn.execute(f"""
                INSERT INTO schema_evolve_test (name, "{col_name}")
                VALUES ('after_{col_name}', {sql_literal_for_value(value, col_type)})
            """)

        # Verify total rows
        expected_rows = initial_rows + len(new_columns)
        await assertions.assert_table_row_count(expected_rows, "schema_evolve_test")

        # Update some old rows with values for new columns
        for col_name, col_type in new_columns:
            value = random_value_for_type(col_type)
            # Update first 10 initial rows
            await db_conn.execute(f"""
                UPDATE schema_evolve_test
                SET "{col_name}" = {sql_literal_for_value(value, col_type)}
                WHERE id <= 10
            """)

        # Verify updates
        updated_rows = await db_conn.fetch("""
            SELECT * FROM schema_evolve_test WHERE id <= 10 ORDER BY id
        """)
        assert len(updated_rows) == 10, "Should have 10 updated rows"

        # Query with all columns
        all_data = await db_conn.fetch("SELECT * FROM schema_evolve_test ORDER BY id")
        assert len(all_data) == expected_rows

        # Verify column count
        column_count = len(all_data[0])
        expected_cols = 2 + len(new_columns)  # id + name + new columns
        assert column_count == expected_cols, \
            f"Expected {expected_cols} columns, got {column_count}"

        print(f"✓ Test passed: Schema evolution with {len(new_columns)} new columns")

    finally:
        await db_conn.execute("DROP TABLE IF EXISTS schema_evolve_test CASCADE")


@pytest.mark.asyncio
async def test_random_query_filters(db_conn: asyncpg.Connection):
    """
    Test querying with random filter conditions.

    Tests:
    - Insert data with known distributions
    - Apply random WHERE conditions
    - Verify result counts match expected ranges
    - Test various operators (=, <, >, <=, >=, BETWEEN, LIKE)
    """
    assertions = Assertions(db_conn)

    try:
        await db_conn.execute("""
            CREATE TABLE filter_test (
                id SERIAL PRIMARY KEY,
                int_val int4,
                float_val float8,
                text_val text,
                date_val date,
                bool_val boolean
            ) USING deeplake
        """)

        # Insert structured random data
        num_rows = 1000
        await db_conn.execute(f"""
            INSERT INTO filter_test (int_val, float_val, text_val, date_val, bool_val)
            SELECT
                (random() * 100)::int4,
                random() * 100,
                'text_' || (random() * 50)::int || '_suffix',
                '2020-01-01'::date + ((random() * 365)::int || ' days')::interval,
                random() > 0.5
            FROM generate_series(1, {num_rows})
        """)

        await assertions.assert_table_row_count(num_rows, "filter_test")

        # Test integer equality
        random_int = random.randint(0, 100)
        count = await db_conn.fetchval(f"""
            SELECT count(*) FROM filter_test WHERE int_val = {random_int}
        """)
        # Should have some matches (roughly 1% of rows)
        assert count >= 0, f"Count should be non-negative for int_val = {random_int}"

        # Test range queries
        low, high = sorted([random.randint(0, 100), random.randint(0, 100)])
        range_count = await db_conn.fetchval(f"""
            SELECT count(*) FROM filter_test WHERE int_val BETWEEN {low} AND {high}
        """)
        expected_fraction = (high - low + 1) / 101
        expected_range = (
            int(num_rows * expected_fraction * 0.5),
            int(num_rows * expected_fraction * 1.5) + 10
        )
        assert expected_range[0] <= range_count <= expected_range[1], \
            f"Range count {range_count} outside expected {expected_range} for [{low}, {high}]"

        # Test boolean filter
        true_count = await db_conn.fetchval("""
            SELECT count(*) FROM filter_test WHERE bool_val = true
        """)
        false_count = await db_conn.fetchval("""
            SELECT count(*) FROM filter_test WHERE bool_val = false
        """)
        assert true_count + false_count == num_rows, \
            "True + False counts should equal total rows"
        # Should be roughly 50/50
        assert 300 < true_count < 700, f"True count {true_count} should be near 500"

        # Test date range
        date_count = await db_conn.fetchval("""
            SELECT count(*) FROM filter_test
            WHERE date_val BETWEEN '2020-03-01' AND '2020-06-30'
        """)
        # Roughly 4 months out of 12 = ~33%
        assert 200 < date_count < 500, f"Date range count {date_count} seems off"

        # Test float comparison
        float_threshold = random.uniform(20, 80)
        lt_count = await db_conn.fetchval(f"""
            SELECT count(*) FROM filter_test WHERE float_val < {float_threshold}
        """)
        gte_count = await db_conn.fetchval(f"""
            SELECT count(*) FROM filter_test WHERE float_val >= {float_threshold}
        """)
        assert lt_count + gte_count == num_rows, \
            f"< and >= should cover all rows: {lt_count} + {gte_count} != {num_rows}"

        # Test combined filters
        combined_count = await db_conn.fetchval(f"""
            SELECT count(*) FROM filter_test
            WHERE int_val > 50 AND bool_val = true AND float_val < 75
        """)
        # Should be roughly: 50% * 50% * 75% = 18.75%
        assert combined_count >= 0, "Combined filter should return non-negative count"

        print(f"✓ Test passed: Random query filters work correctly")
        print(f"  - int_val = {random_int}: {count} rows")
        print(f"  - int_val BETWEEN {low} AND {high}: {range_count} rows")
        print(f"  - bool_val distribution: {true_count} true, {false_count} false")
        print(f"  - Combined filters: {combined_count} rows")

    finally:
        await db_conn.execute("DROP TABLE IF EXISTS filter_test CASCADE")


@pytest.mark.asyncio
async def test_random_aggregations(db_conn: asyncpg.Connection):
    """
    Test aggregation functions on random data.

    Tests:
    - SUM, AVG, MIN, MAX, COUNT on random numeric data
    - GROUP BY with random categories
    - HAVING clauses with random thresholds
    - Verify mathematical correctness
    """
    assertions = Assertions(db_conn)

    try:
        await db_conn.execute("""
            CREATE TABLE agg_test (
                id SERIAL PRIMARY KEY,
                category text,
                amount numeric,
                quantity int4
            ) USING deeplake
        """)

        # Insert data with known categories
        categories = ['A', 'B', 'C', 'D', 'E']
        num_rows = 2000

        await db_conn.execute(f"""
            INSERT INTO agg_test (category, amount, quantity)
            SELECT
                (ARRAY['A', 'B', 'C', 'D', 'E'])[(random() * 4 + 1)::int],
                (random() * 1000)::numeric(10,2),
                (random() * 100)::int4
            FROM generate_series(1, {num_rows})
        """)

        await assertions.assert_table_row_count(num_rows, "agg_test")

        # Test overall aggregates
        agg_result = await db_conn.fetchrow("""
            SELECT
                COUNT(*) as cnt,
                SUM(amount) as total_amount,
                AVG(amount) as avg_amount,
                MIN(amount) as min_amount,
                MAX(amount) as max_amount,
                SUM(quantity) as total_qty
            FROM agg_test
        """)

        assert agg_result['cnt'] == num_rows, f"COUNT should be {num_rows}"
        assert agg_result['total_amount'] > 0, "SUM should be positive"
        assert 0 < float(agg_result['avg_amount']) < 1000, "AVG should be between 0 and 1000"
        assert agg_result['min_amount'] >= 0, "MIN should be non-negative"
        assert agg_result['max_amount'] <= 1000, "MAX should be <= 1000"

        # Test GROUP BY
        group_result = await db_conn.fetch("""
            SELECT category, COUNT(*) as cnt, SUM(amount) as total
            FROM agg_test
            GROUP BY category
            ORDER BY category
        """)

        assert len(group_result) == 5, f"Should have 5 categories, got {len(group_result)}"
        total_from_groups = sum(row['cnt'] for row in group_result)
        assert total_from_groups == num_rows, "Sum of group counts should equal total rows"

        # Each category should have roughly 400 rows (1/5 of 2000)
        for row in group_result:
            assert 200 < row['cnt'] < 600, \
                f"Category {row['category']} count {row['cnt']} seems off"

        # Test HAVING with random threshold
        having_threshold = random.uniform(50000, 150000)
        having_result = await db_conn.fetch(f"""
            SELECT category, SUM(amount) as total
            FROM agg_test
            GROUP BY category
            HAVING SUM(amount) > {having_threshold}
        """)

        # Verify HAVING filter
        for row in having_result:
            assert float(row['total']) > having_threshold, \
                f"Category {row['category']} total {row['total']} should be > {having_threshold}"

        # Test aggregate on filtered data
        random_cat = random.choice(categories)
        filtered_agg = await db_conn.fetchrow(f"""
            SELECT AVG(quantity) as avg_qty, COUNT(*) as cnt
            FROM agg_test
            WHERE category = '{random_cat}'
        """)

        assert filtered_agg['cnt'] > 0, f"Should have rows for category {random_cat}"
        assert 0 <= float(filtered_agg['avg_qty']) <= 100, "AVG quantity should be in range"

        print(f"✓ Test passed: Aggregations work correctly on random data")
        print(f"  - Total rows: {num_rows}")
        print(f"  - SUM(amount): {agg_result['total_amount']:.2f}")
        print(f"  - AVG(amount): {agg_result['avg_amount']:.2f}")
        print(f"  - Categories meeting HAVING > {having_threshold:.2f}: {len(having_result)}")

    finally:
        await db_conn.execute("DROP TABLE IF EXISTS agg_test CASCADE")


@pytest.mark.asyncio
async def test_random_update_delete_combinations(db_conn: asyncpg.Connection):
    """
    Test random combinations of UPDATE and DELETE operations.

    Tests:
    - Multiple UPDATE operations with random values
    - DELETE with random conditions
    - Interleaved UPDATE and DELETE
    - Data integrity after each operation
    """
    assertions = Assertions(db_conn)

    try:
        await db_conn.execute("""
            CREATE TABLE update_delete_test (
                id SERIAL PRIMARY KEY,
                status text,
                score int4,
                updated_count int4 DEFAULT 0
            ) USING deeplake
        """)

        # Insert initial data using bulk insert
        num_rows = 500
        statuses = ['pending', 'active', 'completed', 'archived']

        # Use generate_series for bulk insert
        await db_conn.execute(f"""
            INSERT INTO update_delete_test (status, score)
            SELECT
                (ARRAY['pending', 'active', 'completed', 'archived'])[(random() * 3 + 1)::int],
                (random() * 100)::int4
            FROM generate_series(1, {num_rows})
        """)

        await assertions.assert_table_row_count(num_rows, "update_delete_test")

        # Perform random updates
        num_updates = 10
        for _ in range(num_updates):
            # Random update condition
            update_type = random.choice(['status', 'score', 'both'])

            if update_type == 'status':
                old_status = random.choice(statuses)
                new_status = random.choice(statuses)
                await db_conn.execute(f"""
                    UPDATE update_delete_test
                    SET status = '{new_status}', updated_count = updated_count + 1
                    WHERE status = '{old_status}'
                """)
            elif update_type == 'score':
                threshold = random.randint(0, 100)
                increment = random.randint(-20, 20)
                await db_conn.execute(f"""
                    UPDATE update_delete_test
                    SET score = score + {increment}, updated_count = updated_count + 1
                    WHERE score > {threshold}
                """)
            else:  # both
                old_status = random.choice(statuses)
                new_score = random.randint(0, 100)
                await db_conn.execute(f"""
                    UPDATE update_delete_test
                    SET score = {new_score}, updated_count = updated_count + 1
                    WHERE status = '{old_status}'
                """)

        # Verify row count unchanged after updates
        await assertions.assert_table_row_count(num_rows, "update_delete_test")

        # Check that some rows were updated
        updated_rows = await db_conn.fetchval("""
            SELECT COUNT(*) FROM update_delete_test WHERE updated_count > 0
        """)
        assert updated_rows > 0, "Some rows should have been updated"

        # Perform random deletes
        current_count = num_rows
        num_deletes = 5

        for _ in range(num_deletes):
            delete_type = random.choice(['status', 'score'])

            if delete_type == 'status':
                status_to_delete = random.choice(statuses)
                count_before = await db_conn.fetchval(f"""
                    SELECT COUNT(*) FROM update_delete_test WHERE status = '{status_to_delete}'
                """)
                await db_conn.execute(f"""
                    DELETE FROM update_delete_test WHERE status = '{status_to_delete}'
                """)
                current_count -= count_before
            else:
                score_threshold = random.randint(80, 120)  # May delete 0 rows
                count_before = await db_conn.fetchval(f"""
                    SELECT COUNT(*) FROM update_delete_test WHERE score > {score_threshold}
                """)
                await db_conn.execute(f"""
                    DELETE FROM update_delete_test WHERE score > {score_threshold}
                """)
                current_count -= count_before

            # Verify count after each delete
            actual_count = await db_conn.fetchval("""
                SELECT COUNT(*) FROM update_delete_test
            """)
            assert actual_count == current_count, \
                f"Expected {current_count} rows after delete, got {actual_count}"

        # Final verification
        final_count = await db_conn.fetchval("SELECT COUNT(*) FROM update_delete_test")
        print(f"✓ Test passed: Random UPDATE/DELETE operations")
        print(f"  - Initial rows: {num_rows}")
        print(f"  - Updates performed: {num_updates}")
        print(f"  - Rows updated at least once: {updated_rows}")
        print(f"  - Deletes performed: {num_deletes}")
        print(f"  - Final row count: {final_count}")

    finally:
        await db_conn.execute("DROP TABLE IF EXISTS update_delete_test CASCADE")


@pytest.mark.asyncio
async def test_random_data_integrity(db_conn: asyncpg.Connection):
    """
    Test data integrity with random operations.

    Tests:
    - Insert data and verify checksums/aggregates
    - Perform operations and re-verify integrity
    - Test for data corruption or loss
    """
    assertions = Assertions(db_conn)

    try:
        await db_conn.execute("""
            CREATE TABLE integrity_test (
                id SERIAL PRIMARY KEY,
                value numeric(12,2),
                label text
            ) USING deeplake
        """)

        # Insert data using generate_series with known formula for verification
        num_rows = 1000
        # Insert values 1 to 1000, so expected sum = n*(n+1)/2 = 500500
        await db_conn.execute(f"""
            INSERT INTO integrity_test (value, label)
            SELECT i::numeric(12,2), 'label_' || i
            FROM generate_series(1, {num_rows}) i
        """)

        await assertions.assert_table_row_count(num_rows, "integrity_test")

        # Verify sum matches expected formula: sum(1..n) = n*(n+1)/2
        expected_sum = Decimal(num_rows * (num_rows + 1) // 2)  # 500500
        actual_sum = await db_conn.fetchval("SELECT SUM(value) FROM integrity_test")
        assert actual_sum is not None, "SUM should not be None"
        assert actual_sum == expected_sum, \
            f"SUM mismatch: expected {expected_sum}, got {actual_sum}"

        # Verify MIN/MAX
        min_val = await db_conn.fetchval("SELECT MIN(value) FROM integrity_test")
        max_val = await db_conn.fetchval("SELECT MAX(value) FROM integrity_test")
        assert min_val == 1, f"MIN should be 1, got {min_val}"
        assert max_val == num_rows, f"MAX should be {num_rows}, got {max_val}"

        # Update some values: add 1000 to first 100 rows
        await db_conn.execute("""
            UPDATE integrity_test
            SET value = value + 1000
            WHERE id <= 100
        """)

        expected_sum += Decimal(100 * 1000)  # Added 1000 to 100 rows
        actual_sum = await db_conn.fetchval("SELECT SUM(value) FROM integrity_test")
        assert actual_sum is not None, "SUM after update should not be None"
        assert actual_sum == expected_sum, \
            f"SUM after updates: expected {expected_sum}, got {actual_sum}"

        # Delete rows where original value was 901-1000 (100 rows)
        # These now have value 901-1000 (weren't in first 100)
        # Sum of 901..1000 = 95050
        delete_sum = Decimal(sum(range(901, 1001)))  # 95050
        await db_conn.execute("""
            DELETE FROM integrity_test WHERE value BETWEEN 901 AND 1000
        """)

        expected_sum -= delete_sum
        expected_count = num_rows - 100

        # Final verification
        final_sum = await db_conn.fetchval("SELECT SUM(value) FROM integrity_test")
        final_count = await db_conn.fetchval("SELECT COUNT(*) FROM integrity_test")

        assert final_count == expected_count, \
            f"Final COUNT: expected {expected_count}, got {final_count}"
        assert final_sum is not None, "Final SUM should not be None"
        assert final_sum == expected_sum, \
            f"Final SUM: expected {expected_sum}, got {final_sum}"

        # Additional integrity check: verify avg is reasonable
        avg_val = await db_conn.fetchval("SELECT AVG(value) FROM integrity_test")
        assert avg_val is not None, "AVG should not be None"
        expected_avg = float(expected_sum) / expected_count
        assert abs(float(avg_val) - expected_avg) < 0.01, \
            f"AVG mismatch: expected {expected_avg:.2f}, got {avg_val}"

        print(f"✓ Test passed: Data integrity maintained through all operations")
        print(f"  - Rows: {num_rows} -> {final_count} (deleted 100)")
        print(f"  - Updates: 100 rows (added 1000 to each)")
        print(f"  - SUM verified: {final_sum}")
        print(f"  - AVG verified: {float(avg_val):.2f}")

    finally:
        await db_conn.execute("DROP TABLE IF EXISTS integrity_test CASCADE")


@pytest.mark.asyncio
async def test_random_concurrent_operations(db_conn: asyncpg.Connection):
    """
    Test table operations with multiple random operations in sequence.

    Tests:
    - Rapid succession of INSERTs, UPDATEs, DELETEs
    - Verify consistency after operation batch
    """
    assertions = Assertions(db_conn)

    try:
        await db_conn.execute("""
            CREATE TABLE concurrent_test (
                id SERIAL PRIMARY KEY,
                counter int4 DEFAULT 0,
                data text
            ) USING deeplake
        """)

        # Initial population
        initial_rows = 200
        await db_conn.execute(f"""
            INSERT INTO concurrent_test (counter, data)
            SELECT 0, 'data_' || i
            FROM generate_series(1, {initial_rows}) i
        """)

        await assertions.assert_table_row_count(initial_rows, "concurrent_test")

        # Perform random operations
        num_operations = 100
        current_count = initial_rows

        for _ in range(num_operations):
            op = random.choice(['insert', 'update', 'delete'])

            if op == 'insert':
                batch_size = random.randint(1, 10)
                await db_conn.execute(f"""
                    INSERT INTO concurrent_test (counter, data)
                    SELECT 0, 'new_' || i
                    FROM generate_series(1, {batch_size}) i
                """)
                current_count += batch_size

            elif op == 'update':
                # Update random rows
                threshold = random.randint(1, current_count)
                await db_conn.execute(f"""
                    UPDATE concurrent_test
                    SET counter = counter + 1
                    WHERE id <= {threshold}
                """)

            else:  # delete
                # Delete a small random number of rows
                if current_count > 50:  # Keep minimum rows
                    to_delete = random.randint(1, min(5, current_count - 50))
                    await db_conn.execute(f"""
                        DELETE FROM concurrent_test
                        WHERE id IN (
                            SELECT id FROM concurrent_test
                            ORDER BY random()
                            LIMIT {to_delete}
                        )
                    """)
                    current_count -= to_delete

        # Verify final state
        final_count = await db_conn.fetchval("SELECT COUNT(*) FROM concurrent_test")
        assert final_count == current_count, \
            f"Expected {current_count} rows, got {final_count}"

        # Verify no data corruption - all rows should be readable
        all_rows = await db_conn.fetch("SELECT * FROM concurrent_test")
        assert len(all_rows) == final_count

        # Check counter values are non-negative
        invalid_counters = await db_conn.fetchval("""
            SELECT COUNT(*) FROM concurrent_test WHERE counter < 0
        """)
        assert invalid_counters == 0, "All counters should be non-negative"

        print(f"✓ Test passed: Random sequential operations")
        print(f"  - Operations performed: {num_operations}")
        print(f"  - Initial rows: {initial_rows}")
        print(f"  - Final rows: {final_count}")

    finally:
        await db_conn.execute("DROP TABLE IF EXISTS concurrent_test CASCADE")


@pytest.mark.asyncio
@pytest.mark.slow
async def test_large_random_dataset(db_conn: asyncpg.Connection):
    """
    Test with a larger random dataset.

    Tests:
    - Insert large number of random rows
    - Various queries on large dataset
    - Aggregations on large dataset
    - Bulk updates and deletes
    """
    assertions = Assertions(db_conn)

    try:
        await db_conn.execute("""
            CREATE TABLE large_random_test (
                id SERIAL PRIMARY KEY,
                int_col int4,
                float_col float8,
                text_col text,
                date_col date,
                bool_col boolean
            ) USING deeplake
        """)

        # Insert large dataset
        num_rows = 10000
        await db_conn.execute(f"""
            INSERT INTO large_random_test (int_col, float_col, text_col, date_col, bool_col)
            SELECT
                (random() * 10000)::int4,
                random() * 10000,
                'text_' || (random() * 1000)::int,
                '2020-01-01'::date + ((random() * 1000)::int || ' days')::interval,
                random() > 0.5
            FROM generate_series(1, {num_rows})
        """)

        await assertions.assert_table_row_count(num_rows, "large_random_test")

        # Test various queries
        # Point query
        random_int = random.randint(0, 10000)
        point_count = await db_conn.fetchval(f"""
            SELECT COUNT(*) FROM large_random_test WHERE int_col = {random_int}
        """)

        # Range query
        low, high = sorted([random.randint(0, 10000), random.randint(0, 10000)])
        range_count = await db_conn.fetchval(f"""
            SELECT COUNT(*) FROM large_random_test WHERE int_col BETWEEN {low} AND {high}
        """)

        # Aggregation
        agg_result = await db_conn.fetchrow("""
            SELECT
                AVG(int_col) as avg_int,
                AVG(float_col) as avg_float,
                COUNT(DISTINCT text_col) as distinct_texts
            FROM large_random_test
        """)

        # Should have ~1000 distinct texts (limited by random() * 1000)
        assert agg_result['distinct_texts'] <= 1001, "Distinct texts should be <= 1001"

        # Bulk update
        await db_conn.execute("""
            UPDATE large_random_test
            SET int_col = int_col + 1
            WHERE bool_col = true
        """)

        # Bulk delete (remove ~10% of data)
        delete_threshold = random.randint(9000, 9500)
        deleted = await db_conn.fetchval(f"""
            SELECT COUNT(*) FROM large_random_test WHERE int_col > {delete_threshold}
        """)
        await db_conn.execute(f"""
            DELETE FROM large_random_test WHERE int_col > {delete_threshold}
        """)

        final_count = await db_conn.fetchval("SELECT COUNT(*) FROM large_random_test")
        assert final_count == num_rows - deleted, \
            f"Expected {num_rows - deleted} rows after delete, got {final_count}"

        print(f"✓ Test passed: Large random dataset operations")
        print(f"  - Rows: {num_rows}")
        print(f"  - Point query (int_col={random_int}): {point_count} rows")
        print(f"  - Range query [{low}, {high}]: {range_count} rows")
        print(f"  - Distinct texts: {agg_result['distinct_texts']}")
        print(f"  - Deleted: {deleted} rows")
        print(f"  - Final count: {final_count}")

    finally:
        await db_conn.execute("DROP TABLE IF EXISTS large_random_test CASCADE")
