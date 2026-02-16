"""
Test stateless catalog sync with reserved schema names like "default".

This tests that tables created in schemas with reserved SQL names (like "default")
are properly synced between PostgreSQL instances via the deeplake catalog.

The bug this tests: sync_worker.cpp and table_storage.cpp were generating
unquoted DDL like:
    CREATE TABLE default.table_name (...)
instead of:
    CREATE TABLE "default"."table_name" (...)

This caused syntax errors because "default" is a SQL reserved word.
"""
import pytest

pytestmark = pytest.mark.skip(reason="client-side DDL WAL replay removed; tests need rework")

import asyncpg
import os
import shutil
import subprocess
import time
from pathlib import Path


# Reuse PostgresInstance from test_stateless_multi_instance
from test_stateless_multi_instance import PostgresInstance, PRIMARY_PORT


@pytest.fixture(scope="module")
def secondary_instance(pg_server, pg_config):
    """
    Create a secondary PostgreSQL instance for catalog sync testing.
    Module-scoped to be reused across tests in this file.
    """
    import tempfile
    tmp_path = Path(tempfile.mkdtemp(prefix="deeplake_reserved_schema_"))

    if os.geteuid() == 0:
        user = os.environ.get("USER", "postgres")
        shutil.chown(str(tmp_path), user=user, group=user)
        os.chmod(tmp_path, 0o777)

    data_dir = tmp_path / "pg_data_secondary"
    log_file = tmp_path / "secondary_server.log"

    instance = PostgresInstance(
        install_dir=pg_config["install"],
        data_dir=data_dir,
        port=5434,  # Different port from other tests
        log_file=log_file,
        extension_path=pg_config["extension_path"],
        major_version=pg_config["major_version"],
    )

    instance.initialize(skip_extension_install=True)
    instance.start()

    yield instance

    instance.cleanup()


@pytest.fixture
async def primary_conn(pg_server):
    """Connection to primary instance."""
    user = os.environ.get("USER", "postgres")
    conn = await asyncpg.connect(
        database="postgres",
        user=user,
        host="localhost",
        port=PRIMARY_PORT,
        statement_cache_size=0
    )

    try:
        await conn.execute("DROP EXTENSION IF EXISTS pg_deeplake CASCADE")
        await conn.execute("CREATE EXTENSION pg_deeplake")
        # Clean up any leftover reserved schemas from previous failed test runs
        for schema in ["default", "user", "table"]:
            await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        yield conn
    finally:
        # Best-effort cleanup of reserved schemas on teardown
        for schema in ["default", "user", "table"]:
            try:
                await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
            except Exception:
                pass
        await conn.close()


@pytest.mark.asyncio
async def test_catalog_sync_default_schema(
    primary_conn: asyncpg.Connection,
    secondary_instance: PostgresInstance,
    temp_dir_for_postgres: str,
):
    """
    Test catalog sync with a table in "default" schema.

    This is the core test for the quoting fix:
    1. Instance A creates schema "default" and a table in it
    2. Instance B connects with same root_path
    3. Instance B should auto-discover the table via catalog
    4. The sync should not fail with "syntax error at or near 'default'"
    """
    shared_root_path = temp_dir_for_postgres
    print(f"\n=== Test: Catalog Sync with 'default' Schema ===")
    print(f"Shared root path: {shared_root_path}")

    # Instance A: Create schema and table
    await primary_conn.execute(f"SET deeplake.root_path = '{shared_root_path}'")

    # Create schema with reserved name
    await primary_conn.execute('CREATE SCHEMA "default"')
    print('Instance A: Created schema "default"')

    # Create table in "default" schema
    await primary_conn.execute('''
        CREATE TABLE "default".test_chunks (
            id TEXT,
            content TEXT,
            created_at BIGINT
        ) USING deeplake
    ''')
    print('Instance A: Created table "default".test_chunks')

    # Insert test data
    await primary_conn.execute('''
        INSERT INTO "default".test_chunks VALUES
            ('chunk1', 'This is the first chunk', 1000),
            ('chunk2', 'This is the second chunk', 2000),
            ('chunk3', 'This is the third chunk', 3000)
    ''')
    print("Instance A: Inserted 3 rows")

    # Verify on Instance A
    count_a = await primary_conn.fetchval('SELECT COUNT(*) FROM "default".test_chunks')
    assert count_a == 3, f"Instance A should have 3 rows, got {count_a}"

    # Get catalog entry
    catalog_entry = await primary_conn.fetchrow('''
        SELECT table_name, ds_path FROM pg_deeplake_tables
        WHERE table_name = 'default.test_chunks'
    ''')
    assert catalog_entry is not None, "Table should be in catalog"
    print(f"Instance A: Catalog entry - {catalog_entry['table_name']}: {catalog_entry['ds_path']}")

    # Instance B: Connect and discover table via catalog sync
    print("\n--- Instance B: Connecting and discovering table ---")
    conn_b = await secondary_instance.connect()

    try:
        await conn_b.execute("CREATE EXTENSION IF NOT EXISTS pg_deeplake")


        # This is the critical part - setting root_path triggers catalog sync
        # which should properly quote "default" schema name in generated DDL
        await conn_b.execute(f"SET deeplake.root_path = '{shared_root_path}'")
        print("Instance B: Set root_path (triggers catalog sync)")

        # Verify table was auto-discovered
        catalog_tables = await conn_b.fetch('''
            SELECT table_name, ds_path FROM pg_deeplake_tables
        ''')
        table_names = [t['table_name'] for t in catalog_tables]
        print(f"Instance B: Discovered tables: {table_names}")

        assert 'default.test_chunks' in table_names, \
            f"Table 'default.test_chunks' should be auto-discovered, found: {table_names}"
        print('Instance B: Table "default".test_chunks auto-discovered!')

        # Verify table is visible via pg_tables
        pg_tables = await conn_b.fetch('''
            SELECT schemaname, tablename FROM pg_tables
            WHERE tablename = 'test_chunks'
        ''')
        assert len(pg_tables) == 1, "Table should be visible in pg_tables"
        assert pg_tables[0]['schemaname'] == 'default', \
            f"Schema should be 'default', got {pg_tables[0]['schemaname']}"
        print('Instance B: Table visible in pg_tables with schema "default"')

        # Query data from Instance B
        count_b = await conn_b.fetchval('SELECT COUNT(*) FROM "default".test_chunks')
        assert count_b == 3, f"Instance B should see 3 rows, got {count_b}"
        print(f"Instance B: Row count verified: {count_b}")

        # Verify data contents
        rows = await conn_b.fetch('''
            SELECT id, content FROM "default".test_chunks ORDER BY id
        ''')
        assert len(rows) == 3
        assert rows[0]['id'] == 'chunk1'
        assert rows[1]['id'] == 'chunk2'
        assert rows[2]['id'] == 'chunk3'
        print("Instance B: Data contents verified!")

    finally:
        await conn_b.execute("RESET deeplake.root_path")
        await conn_b.close()

    # Cleanup
    await primary_conn.execute('DROP TABLE IF EXISTS "default".test_chunks CASCADE')
    await primary_conn.execute('DROP SCHEMA IF EXISTS "default" CASCADE')
    await primary_conn.execute("RESET deeplake.root_path")
    print("\n=== Test Passed: Catalog sync with 'default' schema works ===")


@pytest.mark.asyncio
async def test_catalog_sync_multiple_reserved_schemas(
    primary_conn: asyncpg.Connection,
    secondary_instance: PostgresInstance,
    temp_dir_for_postgres: str,
):
    """
    Test catalog sync with multiple reserved-name schemas.

    Tests that various SQL reserved words work as schema names.
    """
    shared_root_path = temp_dir_for_postgres
    reserved_schemas = ["default", "user", "table"]

    print(f"\n=== Test: Catalog Sync with Multiple Reserved Schemas ===")
    print(f"Testing schemas: {reserved_schemas}")

    # Instance A: Create schemas and tables
    await primary_conn.execute(f"SET deeplake.root_path = '{shared_root_path}'")

    for schema in reserved_schemas:
        await primary_conn.execute(f'CREATE SCHEMA "{schema}"')
        await primary_conn.execute(f'''
            CREATE TABLE "{schema}".data (
                id INT,
                value TEXT
            ) USING deeplake
        ''')
        await primary_conn.execute(f'''
            INSERT INTO "{schema}".data VALUES (1, '{schema}_value')
        ''')
        print(f'Instance A: Created "{schema}".data')

    # Instance B: Discover all tables
    conn_b = await secondary_instance.connect()

    try:
        await conn_b.execute("CREATE EXTENSION IF NOT EXISTS pg_deeplake")

        await conn_b.execute(f"SET deeplake.root_path = '{shared_root_path}'")

        # Verify all tables discovered
        catalog_tables = await conn_b.fetch('''
            SELECT table_name FROM pg_deeplake_tables
        ''')
        table_names = [t['table_name'] for t in catalog_tables]
        print(f"Instance B: Discovered tables: {table_names}")

        for schema in reserved_schemas:
            expected_name = f'{schema}.data'
            assert expected_name in table_names, \
                f"Table '{expected_name}' should be discovered, found: {table_names}"

            # Query each table
            count = await conn_b.fetchval(f'SELECT COUNT(*) FROM "{schema}".data')
            assert count == 1, f"Table {schema}.data should have 1 row"

            value = await conn_b.fetchval(f'SELECT value FROM "{schema}".data')
            assert value == f'{schema}_value', f"Wrong value in {schema}.data"

            print(f'Instance B: Verified "{schema}".data')

    finally:
        await conn_b.execute("RESET deeplake.root_path")
        await conn_b.close()

    # Cleanup
    for schema in reserved_schemas:
        await primary_conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
    await primary_conn.execute("RESET deeplake.root_path")
    print("\n=== Test Passed: All reserved-name schemas sync correctly ===")


@pytest.mark.asyncio
async def test_catalog_sync_default_schema_with_indexes(
    primary_conn: asyncpg.Connection,
    secondary_instance: PostgresInstance,
    temp_dir_for_postgres: str,
):
    """
    Test catalog sync with indexed table in "default" schema.

    Verifies that tables with indexes are properly synced.
    """
    shared_root_path = temp_dir_for_postgres
    print(f"\n=== Test: Catalog Sync with Indexed Table in 'default' Schema ===")

    # Instance A: Create schema, table, and indexes
    await primary_conn.execute(f"SET deeplake.root_path = '{shared_root_path}'")
    await primary_conn.execute('CREATE SCHEMA "default"')

    await primary_conn.execute('''
        CREATE TABLE "default".embeddings (
            id TEXT,
            chunk_id TEXT,
            embedding FLOAT[],
            created_at BIGINT
        ) USING deeplake
    ''')
    print('Instance A: Created "default".embeddings')

    # Create indexes
    await primary_conn.execute('''
        CREATE INDEX idx_default_embeddings_id
        ON "default".embeddings USING deeplake_index (id) WITH (index_type='inverted')
    ''')
    await primary_conn.execute('''
        CREATE INDEX idx_default_embeddings_chunk_id
        ON "default".embeddings USING deeplake_index (chunk_id) WITH (index_type='inverted')
    ''')
    print("Instance A: Created indexes")

    # Insert data
    await primary_conn.execute('''
        INSERT INTO "default".embeddings VALUES
            ('emb1', 'chunk1', ARRAY[0.1, 0.2, 0.3], 1000),
            ('emb2', 'chunk2', ARRAY[0.4, 0.5, 0.6], 2000)
    ''')

    # Instance B: Discover and query
    conn_b = await secondary_instance.connect()

    try:
        await conn_b.execute("CREATE EXTENSION IF NOT EXISTS pg_deeplake")

        await conn_b.execute(f"SET deeplake.root_path = '{shared_root_path}'")

        # Verify table discovered
        count = await conn_b.fetchval('SELECT COUNT(*) FROM "default".embeddings')
        assert count == 2, f"Should have 2 rows, got {count}"
        print(f"Instance B: Table discovered with {count} rows")

        # Query using indexed column
        result = await conn_b.fetchval('''
            SELECT chunk_id FROM "default".embeddings WHERE id = 'emb1'
        ''')
        assert result == 'chunk1', f"Expected 'chunk1', got {result}"
        print("Instance B: Index query worked")

    finally:
        await conn_b.execute("RESET deeplake.root_path")
        await conn_b.close()

    # Cleanup
    await primary_conn.execute('DROP SCHEMA IF EXISTS "default" CASCADE')
    await primary_conn.execute("RESET deeplake.root_path")
    print("\n=== Test Passed: Indexed table in 'default' schema syncs correctly ===")


@pytest.mark.asyncio
async def test_catalog_sync_default_schema_write_from_secondary(
    primary_conn: asyncpg.Connection,
    secondary_instance: PostgresInstance,
    temp_dir_for_postgres: str,
):
    """
    Test that secondary instance can write to auto-discovered table in "default" schema.
    """
    shared_root_path = temp_dir_for_postgres
    print(f"\n=== Test: Write to Auto-Discovered Table in 'default' Schema ===")

    # Instance A: Create schema and table
    await primary_conn.execute(f"SET deeplake.root_path = '{shared_root_path}'")
    await primary_conn.execute('CREATE SCHEMA "default"')
    await primary_conn.execute('''
        CREATE TABLE "default".shared_data (
            id INT,
            source TEXT
        ) USING deeplake
    ''')
    await primary_conn.execute('''
        INSERT INTO "default".shared_data VALUES (1, 'instance_a')
    ''')
    print('Instance A: Created "default".shared_data with 1 row')

    # Instance B: Discover and write
    conn_b = await secondary_instance.connect()

    try:
        await conn_b.execute("CREATE EXTENSION IF NOT EXISTS pg_deeplake")

        await conn_b.execute(f"SET deeplake.root_path = '{shared_root_path}'")

        # Insert from Instance B
        await conn_b.execute('''
            INSERT INTO "default".shared_data VALUES (2, 'instance_b')
        ''')
        print("Instance B: Inserted row into auto-discovered table")

        # Verify both rows visible from Instance B
        count_b = await conn_b.fetchval('SELECT COUNT(*) FROM "default".shared_data')
        assert count_b == 2, f"Instance B should see 2 rows, got {count_b}"

        # Verify from Instance A
        count_a = await primary_conn.fetchval('SELECT COUNT(*) FROM "default".shared_data')
        assert count_a == 2, f"Instance A should see 2 rows, got {count_a}"

        print(f"Both instances see {count_a} rows")

        # Verify data sources
        sources = await conn_b.fetch('''
            SELECT DISTINCT source FROM "default".shared_data ORDER BY source
        ''')
        assert len(sources) == 2
        assert sources[0]['source'] == 'instance_a'
        assert sources[1]['source'] == 'instance_b'
        print("Data from both instances verified!")

    finally:
        await conn_b.execute("RESET deeplake.root_path")
        await conn_b.close()

    # Cleanup
    await primary_conn.execute('DROP SCHEMA IF EXISTS "default" CASCADE')
    await primary_conn.execute("RESET deeplake.root_path")
    print("\n=== Test Passed: Write to auto-discovered 'default' schema table works ===")
