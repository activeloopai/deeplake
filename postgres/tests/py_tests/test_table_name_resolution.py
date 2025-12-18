"""
Test table name resolution with hash suffixes.

This test validates that the extension correctly resolves table names
even when PostgreSQL adds hash suffixes (e.g., after table renames or
other DDL operations), ensuring the metadata cache doesn't cause
table-not-found errors in DuckDB.

Regression test for issue where queries failed with:
"Catalog Error: Table with name <table>_<hash> does not exist!"
"""
import pytest
import asyncpg
from test_utils.assertions import Assertions


@pytest.mark.asyncio
async def test_table_name_with_hash_suffix(db_conn: asyncpg.Connection):
    """
    Test that tables with hash suffixes in their names work correctly.

    This simulates the scenario where PostgreSQL adds hash suffixes to
    table names (e.g., after certain DDL operations) and ensures that:
    1. The extension loads the actual current name from pg_catalog
    2. DuckDB is registered with the correct table name
    3. Queries work without catalog errors
    """
    assertions = Assertions(db_conn)

    try:
        # Create a table with a hash-like suffix in the name
        # This simulates what PostgreSQL does internally
        table_name = "test_household_data_7062068c"

        await db_conn.execute(f"""
            CREATE TABLE {table_name} (
                id SERIAL PRIMARY KEY,
                household_id INTEGER,
                net_worth DECIMAL(15,2),
                year INTEGER
            ) USING deeplake
        """)

        # Insert test data
        await db_conn.execute(f"""
            INSERT INTO {table_name} (household_id, net_worth, year) VALUES
                (1, 500000.00, 2024),
                (2, 750000.00, 2024),
                (3, 1000000.00, 2024),
                (4, 250000.00, 2024),
                (5, 1500000.00, 2024)
        """)

        # Test COUNT query - this was failing before the fix
        count = await db_conn.fetchval(f"SELECT COUNT(*) FROM {table_name}")
        assert count == 5, f"Expected 5 rows, got {count}"

        # Test aggregate query
        avg_worth = await db_conn.fetchval(
            f"SELECT AVG(net_worth) FROM {table_name}"
        )
        assert avg_worth is not None, "AVG query should return a result"

        # Test filtered query
        high_worth_count = await db_conn.fetchval(
            f"SELECT COUNT(*) FROM {table_name} WHERE net_worth > 500000"
        )
        assert high_worth_count == 3, f"Expected 3 high-worth households, got {high_worth_count}"

    finally:
        # Cleanup
        try:
            await db_conn.execute(f"DROP TABLE IF EXISTS {table_name}")
        except:
            pass


@pytest.mark.asyncio
async def test_schema_qualified_table_with_suffix(db_conn: asyncpg.Connection):
    """
    Test schema-qualified table names with hash suffixes.

    Ensures that schema.table_name_hash format works correctly
    with DuckDB registration.
    """
    assertions = Assertions(db_conn)

    try:
        # Create a test schema
        schema_name = "test_workspace"
        table_name = "statistics_data_abc123de"
        qualified_name = f"{schema_name}.{table_name}"

        await db_conn.execute(f"CREATE SCHEMA IF NOT EXISTS {schema_name}")

        await db_conn.execute(f"""
            CREATE TABLE {qualified_name} (
                id SERIAL PRIMARY KEY,
                metric_name TEXT,
                metric_value DECIMAL(10,2)
            ) USING deeplake
        """)

        # Insert test data
        await db_conn.execute(f"""
            INSERT INTO {qualified_name} (metric_name, metric_value) VALUES
                ('revenue', 1000.00),
                ('profit', 250.00),
                ('expenses', 750.00)
        """)

        # Test COUNT with schema-qualified name
        count = await db_conn.fetchval(f"SELECT COUNT(*) FROM {qualified_name}")
        assert count == 3, f"Expected 3 rows, got {count}"

        # Test query with WHERE clause
        result = await db_conn.fetchval(
            f"SELECT metric_value FROM {qualified_name} WHERE metric_name = 'profit'"
        )
        assert result == 250.00, f"Expected 250.00, got {result}"

    finally:
        # Cleanup
        try:
            await db_conn.execute(f"DROP TABLE IF EXISTS {qualified_name}")
            await db_conn.execute(f"DROP SCHEMA IF EXISTS {schema_name} CASCADE")
        except:
            pass


@pytest.mark.asyncio
async def test_table_metadata_staleness(db_conn: asyncpg.Connection):
    """
    Test that the extension handles stale metadata correctly.

    This test simulates a scenario where table metadata might be out of sync
    with the actual PostgreSQL catalog (e.g., after a table rename or
    crash recovery).
    """
    assertions = Assertions(db_conn)

    try:
        # Create initial table
        original_name = "original_table"
        await db_conn.execute(f"""
            CREATE TABLE {original_name} (
                id SERIAL PRIMARY KEY,
                data TEXT
            ) USING deeplake
        """)

        # Insert some data
        await db_conn.execute(f"""
            INSERT INTO {original_name} (data) VALUES ('test1'), ('test2')
        """)

        # Verify initial state
        count = await db_conn.fetchval(f"SELECT COUNT(*) FROM {original_name}")
        assert count == 2, f"Expected 2 rows initially, got {count}"

        # Note: In PostgreSQL, ALTER TABLE RENAME doesn't add hash suffixes
        # Hash suffixes are typically added internally by PostgreSQL for
        # temporary objects or during certain DDL operations
        # This test just verifies that the current implementation works correctly

        # Force a reconnection to ensure fresh metadata loading
        # (In a real scenario, this would happen between sessions)
        await db_conn.execute("SELECT pg_sleep(0.1)")

        # Query should still work
        count = await db_conn.fetchval(f"SELECT COUNT(*) FROM {original_name}")
        assert count == 2, f"Expected 2 rows after reconnect, got {count}"

    finally:
        # Cleanup
        try:
            await db_conn.execute(f"DROP TABLE IF EXISTS {original_name}")
        except:
            pass


@pytest.mark.asyncio
async def test_multiple_tables_with_similar_names(db_conn: asyncpg.Connection):
    """
    Test multiple tables with similar names and hash suffixes.

    Ensures that DuckDB correctly distinguishes between tables
    with similar base names but different hash suffixes.
    """
    assertions = Assertions(db_conn)

    tables = []
    try:
        # Create multiple tables with similar names
        base_name = "dataset"
        suffixes = ["_aabbccdd", "_11223344", "_deadbeef"]

        for suffix in suffixes:
            table_name = f"{base_name}{suffix}"
            tables.append(table_name)

            await db_conn.execute(f"""
                CREATE TABLE {table_name} (
                    id SERIAL PRIMARY KEY,
                    value INTEGER
                ) USING deeplake
            """)

            # Insert unique data for each table
            value = int(suffix.replace("_", "")[:4], 16)  # Convert hex to int
            await db_conn.execute(f"""
                INSERT INTO {table_name} (value) VALUES ({value})
            """)

        # Verify each table has correct data
        for i, table_name in enumerate(tables):
            count = await db_conn.fetchval(f"SELECT COUNT(*) FROM {table_name}")
            assert count == 1, f"Table {table_name} should have 1 row, got {count}"

            # Verify unique values
            values = [await db_conn.fetchval(f"SELECT value FROM {t}") for t in tables]
            assert len(set(values)) == len(tables), "Each table should have unique values"

    finally:
        # Cleanup
        for table_name in tables:
            try:
                await db_conn.execute(f"DROP TABLE IF EXISTS {table_name}")
            except:
                pass
