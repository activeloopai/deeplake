"""
Test stateless behavior of pg-deeplake extension with multiple PostgreSQL instances.

This test demonstrates that the pg-deeplake extension can share data between
multiple independent PostgreSQL instances through a shared deeplake.root_path.
This validates the stateless architecture where:
- Table metadata is stored in a catalog at the root_path
- Multiple instances can discover and use tables created by other instances
- Data synchronization works out of the box
"""
import pytest
import asyncpg
import asyncio
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, Optional

# Default port for primary instance (same as conftest.py)
PRIMARY_PORT = 5432


class PostgresInstance:
    """Manages a PostgreSQL instance lifecycle."""

    def __init__(
        self,
        install_dir: Path,
        data_dir: Path,
        port: int,
        log_file: Path,
        extension_path: Path,
        major_version: int = 18,
    ):
        self.install_dir = install_dir
        self.data_dir = data_dir
        self.port = port
        self.log_file = log_file
        self.extension_path = extension_path
        self.major_version = major_version
        self.pg_ctl = install_dir / "bin" / "pg_ctl"
        self.initdb = install_dir / "bin" / "initdb"
        self.user = os.environ.get("USER", "postgres")
        self._started = False

    def _run_cmd(self, cmd: str, check: bool = True) -> subprocess.CompletedProcess:
        """Run a command, handling root vs non-root execution."""
        if os.geteuid() == 0:  # Running as root
            result = subprocess.run(
                ["su", "-", self.user, "-c", cmd],
                capture_output=True,
                text=True
            )
        else:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True
            )
        if check and result.returncode != 0:
            raise RuntimeError(f"Command failed: {cmd}\nstderr: {result.stderr}")
        return result

    def is_running(self) -> bool:
        """Check if PostgreSQL server is running."""
        result = self._run_cmd(f"{self.pg_ctl} status -D {self.data_dir}", check=False)
        return result.returncode == 0

    def stop(self) -> None:
        """Stop PostgreSQL server."""
        if self.is_running():
            self._run_cmd(f"{self.pg_ctl} stop -D {self.data_dir} -m fast", check=False)
            time.sleep(2)
        self._started = False

    def _install_extension(self) -> None:
        """Install PostgreSQL extension files."""
        ext_dir = self.install_dir / "share" / "extension"
        lib_dir = self.install_dir / "lib"

        import platform
        lib_suffix = ".dylib" if platform.system() == "Darwin" else ".so"

        ext_dir.mkdir(parents=True, exist_ok=True)

        # Copy extension files
        for control_file in self.extension_path.glob("*.control"):
            shutil.copy(control_file, ext_dir)

        for sql_file in self.extension_path.glob("*.sql"):
            if sql_file.name != "utils.psql":
                shutil.copy(sql_file, ext_dir)

        # Copy shared library
        lib_file = self.extension_path / f"pg_deeplake_{self.major_version}{lib_suffix}"
        if lib_file.exists():
            shutil.copy(lib_file, lib_dir / f"pg_deeplake{lib_suffix}")
        else:
            raise FileNotFoundError(f"Extension library not found: {lib_file}")

    def initialize(self, skip_extension_install: bool = False) -> None:
        """Initialize database cluster and optionally install extension."""
        # Stop if running
        self.stop()

        # Remove existing data directory
        if self.data_dir.exists():
            shutil.rmtree(self.data_dir)

        # Note: Do NOT create data_dir here - initdb expects to create it.
        # Just ensure the parent directory exists and is accessible.
        parent_dir = self.data_dir.parent
        if not parent_dir.exists():
            parent_dir.mkdir(parents=True, exist_ok=True)
        if os.geteuid() == 0:
            shutil.chown(str(parent_dir), user=self.user, group=self.user)
            os.chmod(parent_dir, 0o777)

        # Install extension only if not already installed by primary
        if not skip_extension_install:
            self._install_extension()

        # Initialize database cluster
        self._run_cmd(f"{self.initdb} -D {self.data_dir} -U {self.user}")

        # Configure shared_preload_libraries and port
        with open(self.data_dir / "postgresql.conf", "a") as f:
            f.write(f"\nport = {self.port}\n")
            f.write("shared_preload_libraries = 'pg_deeplake'\n")
            f.write("max_connections = 100\n")
            f.write("shared_buffers = 64MB\n")

    def start(self) -> None:
        """Start PostgreSQL server."""
        if self.is_running():
            return

        env = os.environ.copy()
        lib_path = str(self.install_dir / "lib")
        ld_library_path = f"{lib_path}:{env.get('LD_LIBRARY_PATH', '')}"

        if os.geteuid() == 0:
            subprocess.run(
                ["su", "-", self.user, "-c",
                 f"LD_LIBRARY_PATH={ld_library_path} {self.pg_ctl} -D {self.data_dir} -l {self.log_file} start"],
                check=True,
            )
        else:
            env["LD_LIBRARY_PATH"] = ld_library_path
            subprocess.run(
                [str(self.pg_ctl), "-D", str(self.data_dir), "-l", str(self.log_file), "start"],
                check=True,
                env=env
            )

        # Wait for server to be ready
        time.sleep(3)

        if not self.is_running():
            raise RuntimeError(f"Failed to start PostgreSQL on port {self.port}")

        self._started = True
        print(f"PostgreSQL instance started on port {self.port}")

    async def connect(self, database: str = "postgres") -> asyncpg.Connection:
        """Create a connection to this instance."""
        return await asyncpg.connect(
            database=database,
            user=self.user,
            host="localhost",
            port=self.port,
            statement_cache_size=0
        )

    def cleanup(self) -> None:
        """Stop server and remove data directory."""
        self.stop()
        if self.data_dir.exists():
            shutil.rmtree(self.data_dir)


@pytest.fixture(scope="session")
def pg_paths(pg_config) -> Dict[str, Path]:
    """Get paths needed for creating additional instances."""
    return {
        "install_dir": pg_config["install"],
        "extension_path": pg_config["extension_path"],
        "major_version": pg_config["major_version"],
    }


@pytest.fixture
async def primary_conn(pg_server):
    """
    Create a connection to the primary instance without loading utility functions.

    This is simpler than primary_conn and avoids potential crashes from utility functions.
    """
    user = os.environ.get("USER", "postgres")
    conn = await asyncpg.connect(
        database="postgres",
        user=user,
        host="localhost",
        port=PRIMARY_PORT,
        statement_cache_size=0
    )

    try:
        # Setup: Clean extension state
        await conn.execute("DROP EXTENSION IF EXISTS pg_deeplake CASCADE")
        await conn.execute("CREATE EXTENSION pg_deeplake")
        yield conn
    finally:
        await conn.close()


@pytest.fixture(scope="session")
def second_instance(pg_server, pg_paths) -> PostgresInstance:
    """
    Create a second PostgreSQL instance on a different port.

    This instance:
    - Runs on port 5433 (vs 5432 for primary)
    - Has its own data directory
    - Uses pg_deeplake extension already installed by primary instance
    - Session-scoped for performance (reused across tests)

    Note: Depends on pg_server to ensure primary is started first and
    extension is properly installed before we initialize.
    """
    import tempfile
    tmp_path = Path(tempfile.mkdtemp(prefix="deeplake_secondary_"))

    # When running as root in CI, ensure the postgres user can access the directory
    if os.geteuid() == 0:
        user = os.environ.get("USER", "postgres")
        shutil.chown(str(tmp_path), user=user, group=user)
        os.chmod(tmp_path, 0o777)

    data_dir = tmp_path / "pg_data_secondary"
    log_file = tmp_path / "secondary_server.log"

    instance = PostgresInstance(
        install_dir=pg_paths["install_dir"],
        data_dir=data_dir,
        port=5433,
        log_file=log_file,
        extension_path=pg_paths["extension_path"],
        major_version=pg_paths["major_version"],
    )

    # Skip extension install - already done by primary instance (pg_server fixture)
    instance.initialize(skip_extension_install=True)
    instance.start()

    yield instance

    instance.cleanup()


@pytest.mark.asyncio
async def test_stateless_data_sync_between_instances(
    primary_conn: asyncpg.Connection,
    second_instance: PostgresInstance,
    temp_dir_for_postgres: str,
):
    """
    Test that data created in one instance is visible from another instance.

    This test:
    1. Instance A creates a table with data at shared root_path
    2. Instance B connects with same root_path
    3. Instance B should see the table via catalog discovery
    4. Both instances can read the data
    """
    shared_root_path = temp_dir_for_postgres
    print(f"\n=== Test: Stateless Data Sync ===")
    print(f"Shared root path: {shared_root_path}")

    # Instance A (primary): Create table and insert data
    print("\n--- Instance A (port 5432): Creating table and inserting data ---")
    await primary_conn.execute(f"SET deeplake.root_path = '{shared_root_path}'")

    await primary_conn.execute("""
        CREATE TABLE stateless_test (
            id INT,
            name TEXT,
            value FLOAT
        ) USING deeplake
    """)
    print("Created table 'stateless_test'")

    # Insert test data
    await primary_conn.execute("""
        INSERT INTO stateless_test VALUES
        (1, 'alice', 100.5),
        (2, 'bob', 200.75),
        (3, 'charlie', 300.25)
    """)
    print("Inserted 3 rows")

    # Verify data in Instance A
    count_a = await primary_conn.fetchval("SELECT COUNT(*) FROM stateless_test")
    assert count_a == 3, f"Instance A should have 3 rows, got {count_a}"
    print(f"Instance A row count: {count_a}")

    # Get the dataset path
    ds_path = await primary_conn.fetchval("""
        SELECT ds_path FROM pg_deeplake_tables
        WHERE table_name = 'public.stateless_test'
    """)
    print(f"Dataset path: {ds_path}")

    # Instance B (secondary): Connect and verify data is visible
    print("\n--- Instance B (port 5433): Connecting and verifying data ---")
    conn_b = await second_instance.connect()

    try:
        # Setup extension (create if not exists for session-scoped instance reuse)
        await conn_b.execute("CREATE EXTENSION IF NOT EXISTS pg_deeplake")

        # Setting root_path should automatically discover and register tables from catalog
        await conn_b.execute(f"SET deeplake.root_path = '{shared_root_path}'")
        print("Instance B: Extension loaded, root_path set")

        # Tables should be automatically discovered from the catalog at root_path
        # No need to manually CREATE TABLE - they should appear after SET root_path
        catalog_tables = await conn_b.fetch("""
            SELECT table_name, ds_path FROM pg_deeplake_tables
        """)
        print(f"Instance B catalog tables: {len(catalog_tables)}")

        # The table created by Instance A should now be visible
        assert len(catalog_tables) >= 1, "Table should be auto-discovered from catalog"

        table_names = [t['table_name'] for t in catalog_tables]
        assert 'public.stateless_test' in table_names, \
            f"stateless_test should be in catalog, found: {table_names}"
        print("Instance B: Table auto-discovered from catalog!")

        # Verify table is visible via \dt (pg_tables)
        pg_tables = await conn_b.fetch("""
            SELECT schemaname, tablename FROM pg_tables
            WHERE tablename = 'stateless_test'
        """)
        assert len(pg_tables) == 1, "Table should be visible in pg_tables (\\dt)"
        assert pg_tables[0]['schemaname'] == 'public'
        assert pg_tables[0]['tablename'] == 'stateless_test'
        print("Instance B: Table visible via \\dt (pg_tables)")

        # Query data from Instance B
        count_b = await conn_b.fetchval("SELECT COUNT(*) FROM stateless_test")
        assert count_b == 3, f"Instance B should see 3 rows, got {count_b}"
        print(f"Instance B row count: {count_b}")

        # Verify data contents match
        rows_b = await conn_b.fetch("SELECT id, name, value FROM stateless_test ORDER BY id")
        expected = [(1, 'alice', 100.5), (2, 'bob', 200.75), (3, 'charlie', 300.25)]

        for i, (row, exp) in enumerate(zip(rows_b, expected)):
            assert row['id'] == exp[0], f"Row {i} id mismatch"
            assert row['name'] == exp[1], f"Row {i} name mismatch"
            assert abs(row['value'] - exp[2]) < 0.001, f"Row {i} value mismatch"
        print("Instance B: All data verified correctly!")

    finally:
        # No need to DROP on instance B - table was auto-discovered, not created locally
        await conn_b.execute("RESET deeplake.root_path")
        await conn_b.close()

    # Cleanup Instance A (this is where the table was actually created)
    await primary_conn.execute("DROP TABLE IF EXISTS stateless_test CASCADE")
    await primary_conn.execute("RESET deeplake.root_path")
    print("\n=== Test Passed: Data sync works between instances ===")


@pytest.mark.asyncio
async def test_stateless_concurrent_writes(
    primary_conn: asyncpg.Connection,
    second_instance: PostgresInstance,
    temp_dir_for_postgres: str,
):
    """
    Test that both instances can write to shared tables.

    This test:
    1. Creates a shared table from Instance A
    2. Both instances insert data concurrently
    3. Both instances should see all the data
    """
    shared_root_path = temp_dir_for_postgres
    print(f"\n=== Test: Concurrent Writes ===")
    print(f"Shared root path: {shared_root_path}")

    # Instance A: Create table
    await primary_conn.execute(f"SET deeplake.root_path = '{shared_root_path}'")
    await primary_conn.execute("""
        CREATE TABLE concurrent_test (
            id INT,
            source TEXT
        ) USING deeplake
    """)
    print("Created shared table 'concurrent_test'")

    # Instance B: Connect and discover table via root_path
    conn_b = await second_instance.connect()
    try:
        await conn_b.execute("CREATE EXTENSION IF NOT EXISTS pg_deeplake")

        # Setting root_path should auto-discover tables from deeplake catalog
        await conn_b.execute(f"SET deeplake.root_path = '{shared_root_path}'")

        # Verify table was auto-discovered
        catalog_tables = await conn_b.fetch("""
            SELECT table_name FROM pg_deeplake_tables
        """)
        table_names = [t['table_name'] for t in catalog_tables]
        assert 'public.concurrent_test' in table_names, \
            f"concurrent_test should be auto-discovered, found: {table_names}"

        # Verify table is visible via \dt (pg_tables)
        pg_tables = await conn_b.fetch("""
            SELECT schemaname, tablename FROM pg_tables
            WHERE tablename = 'concurrent_test'
        """)
        assert len(pg_tables) == 1, "Table should be visible in pg_tables (\\dt)"
        print("Instance B: Table visible via \\dt")

        # Insert from Instance A
        print("Instance A: Inserting data...")
        await primary_conn.execute("""
            INSERT INTO concurrent_test VALUES
            (1, 'instance_a'),
            (2, 'instance_a'),
            (3, 'instance_a')
        """)

        # Insert from Instance B
        print("Instance B: Inserting data...")
        await conn_b.execute("""
            INSERT INTO concurrent_test VALUES
            (4, 'instance_b'),
            (5, 'instance_b'),
            (6, 'instance_b')
        """)

        # Verify total count from both instances
        count_a = await primary_conn.fetchval("SELECT COUNT(*) FROM concurrent_test")
        count_b = await conn_b.fetchval("SELECT COUNT(*) FROM concurrent_test")

        print(f"Instance A sees: {count_a} rows")
        print(f"Instance B sees: {count_b} rows")

        # Both should see all 6 rows (after sync)
        assert count_a == 6, f"Instance A should see 6 rows, got {count_a}"
        assert count_b == 6, f"Instance B should see 6 rows, got {count_b}"

        # Verify data from both sources is present
        sources_a = await primary_conn.fetch("SELECT DISTINCT source FROM concurrent_test ORDER BY source")
        sources_b = await conn_b.fetch("SELECT DISTINCT source FROM concurrent_test ORDER BY source")

        assert len(sources_a) == 2, "Should have data from both instances"
        assert len(sources_b) == 2, "Should have data from both instances"
        print("Both instances see data from both sources!")

    finally:
        # No need to DROP on instance B - table was auto-discovered
        await conn_b.execute("RESET deeplake.root_path")
        await conn_b.close()

    # Cleanup on Instance A (where table was created)
    await primary_conn.execute("DROP TABLE IF EXISTS concurrent_test CASCADE")
    await primary_conn.execute("RESET deeplake.root_path")
    print("\n=== Test Passed: Concurrent writes work ===")


@pytest.mark.asyncio
async def test_stateless_instance_restart_persistence(
    primary_conn: asyncpg.Connection,
    pg_paths: Dict[str, Path],
    temp_dir_for_postgres: str,
    tmp_path: Path,
):
    """
    Test that data persists when a new instance connects to existing data.

    This simulates:
    1. Instance A creates data and disconnects
    2. A brand new Instance B starts and points to same root_path
    3. Instance B should discover and access all the data
    """
    shared_root_path = temp_dir_for_postgres
    print(f"\n=== Test: Instance Restart Persistence ===")
    print(f"Shared root path: {shared_root_path}")

    # Instance A: Create table and data
    await primary_conn.execute(f"SET deeplake.root_path = '{shared_root_path}'")
    await primary_conn.execute("""
        CREATE TABLE persist_test (
            id INT,
            data TEXT
        ) USING deeplake
    """)

    await primary_conn.execute("""
        INSERT INTO persist_test VALUES
        (1, 'persistent_data_1'),
        (2, 'persistent_data_2'),
        (3, 'persistent_data_3'),
        (4, 'persistent_data_4'),
        (5, 'persistent_data_5')
    """)

    print(f"Instance A: Created table with 5 rows")

    # Create a fresh Instance B (simulating a new server start)
    # When running as root in CI, ensure the postgres user can access tmp_path
    if os.geteuid() == 0:
        user = os.environ.get("USER", "postgres")
        shutil.chown(str(tmp_path), user=user, group=user)
        os.chmod(tmp_path, 0o755)

    data_dir_b = tmp_path / "pg_data_fresh"
    log_file_b = tmp_path / "fresh_server.log"

    instance_b = PostgresInstance(
        install_dir=pg_paths["install_dir"],
        data_dir=data_dir_b,
        port=5434,  # Different port
        log_file=log_file_b,
        extension_path=pg_paths["extension_path"],
        major_version=pg_paths["major_version"],
    )

    try:
        # Skip extension install - already done by primary instance
        instance_b.initialize(skip_extension_install=True)
        instance_b.start()
        print("Instance B: Fresh instance started on port 5434")

        conn_b = await instance_b.connect()
        try:
            # Setup extension and point to shared root_path
            await conn_b.execute("CREATE EXTENSION pg_deeplake")

            # Setting root_path should auto-discover tables from deeplake catalog
            await conn_b.execute(f"SET deeplake.root_path = '{shared_root_path}'")

            # Verify table was auto-discovered
            catalog_tables = await conn_b.fetch("""
                SELECT table_name FROM pg_deeplake_tables
            """)
            table_names = [t['table_name'] for t in catalog_tables]
            assert 'public.persist_test' in table_names, \
                f"persist_test should be auto-discovered, found: {table_names}"
            print("Instance B: Table auto-discovered from catalog!")

            # Verify table is visible via \dt (pg_tables)
            pg_tables = await conn_b.fetch("""
                SELECT schemaname, tablename FROM pg_tables
                WHERE tablename = 'persist_test'
            """)
            assert len(pg_tables) == 1, "Table should be visible in pg_tables (\\dt)"
            print("Instance B: Table visible via \\dt")

            # Verify data is accessible (no CREATE TABLE needed!)
            count = await conn_b.fetchval("SELECT COUNT(*) FROM persist_test")
            assert count == 5, f"Fresh instance should see 5 rows, got {count}"
            print(f"Instance B: Sees {count} rows from pre-existing data")

            # Verify actual data
            rows = await conn_b.fetch("SELECT id, data FROM persist_test ORDER BY id")
            for i, row in enumerate(rows, 1):
                assert row['id'] == i
                assert row['data'] == f'persistent_data_{i}'
            print("Instance B: All data verified!")

        finally:
            await conn_b.execute("RESET deeplake.root_path")
            await conn_b.close()

    finally:
        instance_b.cleanup()

    # Cleanup primary - check if connection is still valid
    if not primary_conn.is_closed():
        await primary_conn.execute("DROP TABLE IF EXISTS persist_test CASCADE")
        await primary_conn.execute("RESET deeplake.root_path")
    print("\n=== Test Passed: Data persists across instance restarts ===")


@pytest.mark.asyncio
async def test_stateless_multiple_tables_discovery(
    primary_conn: asyncpg.Connection,
    second_instance: PostgresInstance,
    temp_dir_for_postgres: str,
):
    """
    Test that multiple tables created on one instance are all discoverable.

    This tests the catalog functionality for listing all tables.
    """
    shared_root_path = temp_dir_for_postgres
    print(f"\n=== Test: Multiple Tables Discovery ===")
    print(f"Shared root path: {shared_root_path}")

    # Instance A: Create multiple tables
    await primary_conn.execute(f"SET deeplake.root_path = '{shared_root_path}'")

    table_names = ['users', 'orders', 'products']

    for table in table_names:
        await primary_conn.execute(f"""
            CREATE TABLE {table} (
                id INT,
                name TEXT
            ) USING deeplake
        """)
        await primary_conn.execute(f"INSERT INTO {table} VALUES (1, '{table}_data')")
        print(f"Created table '{table}'")

    # Instance B: Discover and access all tables via root_path
    conn_b = await second_instance.connect()
    try:
        await conn_b.execute("CREATE EXTENSION IF NOT EXISTS pg_deeplake")

        # Setting root_path should auto-discover ALL tables from deeplake catalog
        await conn_b.execute(f"SET deeplake.root_path = '{shared_root_path}'")

        # Verify all tables were auto-discovered
        catalog_tables = await conn_b.fetch("""
            SELECT table_name FROM pg_deeplake_tables
        """)
        discovered_names = [t['table_name'] for t in catalog_tables]

        for table in table_names:
            assert f'public.{table}' in discovered_names, \
                f"Table {table} should be auto-discovered, found: {discovered_names}"

        print(f"Instance B: All {len(table_names)} tables auto-discovered!")

        # Verify all tables are visible via \dt (pg_tables)
        pg_tables = await conn_b.fetch("""
            SELECT tablename FROM pg_tables
            WHERE tablename IN ('users', 'orders', 'products')
        """)
        assert len(pg_tables) == 3, f"All 3 tables should be visible in pg_tables (\\dt), found {len(pg_tables)}"
        print("Instance B: All tables visible via \\dt")

        # Verify all tables are accessible and have correct data (no CREATE TABLE needed!)
        for table in table_names:
            count = await conn_b.fetchval(f"SELECT COUNT(*) FROM {table}")
            assert count == 1, f"Table {table} should have 1 row"
            data = await conn_b.fetchval(f"SELECT name FROM {table}")
            assert data == f'{table}_data', f"Table {table} has wrong data"
            print(f"Instance B: Table '{table}' verified")

        print("All tables discovered and verified!")

    finally:
        await conn_b.execute("RESET deeplake.root_path")
        await conn_b.close()

    # Cleanup on Instance A
    for table in table_names:
        await primary_conn.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
    await primary_conn.execute("RESET deeplake.root_path")
    print("\n=== Test Passed: Multiple tables discovery works ===")


@pytest.mark.asyncio
async def test_stateless_root_path_idempotent(
    primary_conn: asyncpg.Connection,
    temp_dir_for_postgres: str,
):
    """
    Test that setting deeplake.root_path is idempotent.

    Setting the same root_path multiple times should:
    - Not cause errors
    - Not create duplicate table entries
    - Maintain correct table visibility
    """
    shared_root_path = temp_dir_for_postgres
    print(f"\n=== Test: Root Path Idempotency ===")
    print(f"Shared root path: {shared_root_path}")

    # Set root_path and create tables
    await primary_conn.execute(f"SET deeplake.root_path = '{shared_root_path}'")
    await primary_conn.execute("""
        CREATE TABLE idempotent_test (id INT, data TEXT) USING deeplake
    """)
    await primary_conn.execute("INSERT INTO idempotent_test VALUES (1, 'test')")
    print("Created table 'idempotent_test'")

    # Verify initial state
    count1 = await primary_conn.fetchval("""
        SELECT COUNT(*) FROM pg_deeplake_tables WHERE table_name = 'public.idempotent_test'
    """)
    assert count1 == 1, f"Should have exactly 1 catalog entry, got {count1}"
    print(f"Initial catalog entries: {count1}")

    # Set the SAME root_path again multiple times
    for i in range(3):
        await primary_conn.execute(f"SET deeplake.root_path = '{shared_root_path}'")
        print(f"Set root_path again (iteration {i + 1})")

    # Verify no duplicate entries
    count2 = await primary_conn.fetchval("""
        SELECT COUNT(*) FROM pg_deeplake_tables WHERE table_name = 'public.idempotent_test'
    """)
    assert count2 == 1, f"Should still have exactly 1 catalog entry after re-setting, got {count2}"
    print(f"Catalog entries after re-setting: {count2}")

    # Verify table is still accessible
    data = await primary_conn.fetchval("SELECT data FROM idempotent_test WHERE id = 1")
    assert data == 'test', f"Data should be 'test', got {data}"
    print("Table still accessible with correct data")

    # Verify pg_tables has exactly one entry
    pg_count = await primary_conn.fetchval("""
        SELECT COUNT(*) FROM pg_tables WHERE tablename = 'idempotent_test'
    """)
    assert pg_count == 1, f"Should have exactly 1 pg_tables entry, got {pg_count}"
    print(f"pg_tables entries: {pg_count}")

    # Test RESET and re-SET
    await primary_conn.execute("RESET deeplake.root_path")
    print("Reset root_path")

    await primary_conn.execute(f"SET deeplake.root_path = '{shared_root_path}'")
    print("Set root_path again after reset")

    # Should still work correctly
    count3 = await primary_conn.fetchval("""
        SELECT COUNT(*) FROM pg_deeplake_tables WHERE table_name = 'public.idempotent_test'
    """)
    assert count3 == 1, f"Should have exactly 1 catalog entry after reset/re-set, got {count3}"

    data2 = await primary_conn.fetchval("SELECT data FROM idempotent_test WHERE id = 1")
    assert data2 == 'test', f"Data should still be 'test' after reset/re-set, got {data2}"
    print("Table still accessible after reset/re-set")

    # Cleanup
    await primary_conn.execute("DROP TABLE IF EXISTS idempotent_test CASCADE")
    await primary_conn.execute("RESET deeplake.root_path")
    print("\n=== Test Passed: Root path setting is idempotent ===")
