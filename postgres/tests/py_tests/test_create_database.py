"""
Test CREATE DATABASE with pg_deeplake extension.

Verifies that pg_deeplake can be installed and used in newly created databases,
and that DROP DATABASE works cleanly with the extension present.

The CREATE DATABASE post-hook queues the database name into shared memory, and
the background sync worker installs pg_deeplake on its next poll cycle (default
2s). Tests that verify this async behaviour poll pg_extension without a manual
fallback and assert the extension appears within a bounded timeout.

Cross-instance tests verify that database entries written to the shared catalog
by one instance are picked up by the sync worker on a second instance, which
then creates the database locally and installs the extension.
"""
import asyncio
import os
import shutil
import subprocess
import tempfile
import time
import pytest
import asyncpg
from pathlib import Path


SECOND_INSTANCE_PORT = 5434


async def connect_postgres(port=5432):
    """Connect to the default postgres database."""
    user = os.environ.get("USER", "postgres")
    return await asyncpg.connect(
        database="postgres",
        user=user,
        host="localhost",
        port=port,
        statement_cache_size=0,
    )


async def connect_database(dbname, port=5432):
    """Connect to a specific database."""
    user = os.environ.get("USER", "postgres")
    return await asyncpg.connect(
        database=dbname,
        user=user,
        host="localhost",
        port=port,
        statement_cache_size=0,
    )


async def ensure_extension(conn, timeout=5.0):
    """Wait for pg_deeplake extension (async install by sync worker), installing manually if timeout expires."""
    import asyncio
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        ext = await conn.fetchval(
            "SELECT extname FROM pg_extension WHERE extname = 'pg_deeplake'"
        )
        if ext == "pg_deeplake":
            return
        await asyncio.sleep(0.3)
    # Fallback: install manually if the sync worker hasn't picked it up yet
    await conn.execute("CREATE EXTENSION IF NOT EXISTS pg_deeplake")


@pytest.mark.asyncio
async def test_create_database_auto_installs_extension(pg_server):
    """
    Verify that pg_deeplake extension can be installed in a new database.

    CREATE DATABASE followed by CREATE EXTENSION IF NOT EXISTS.
    The hook may auto-install the extension; if not, we install it manually.
    """
    conn = await connect_postgres()
    target_conn = None
    try:
        await conn.execute("DROP DATABASE IF EXISTS test_auto_ext_db")
        await conn.execute("CREATE DATABASE test_auto_ext_db")

        target_conn = await connect_database("test_auto_ext_db")
        await ensure_extension(target_conn)

        ext = await target_conn.fetchval(
            "SELECT extname FROM pg_extension WHERE extname = 'pg_deeplake'"
        )
        assert ext == "pg_deeplake", (
            f"Expected pg_deeplake extension to be installed, got: {ext}"
        )
    finally:
        if target_conn is not None:
            await target_conn.close()
        await conn.execute("DROP DATABASE IF EXISTS test_auto_ext_db")
        await conn.close()


@pytest.mark.asyncio
async def test_create_database_deeplake_works(pg_server):
    """
    Verify that deeplake storage works in a newly created database.

    Steps:
    - CREATE DATABASE test_dl_works_db
    - Connect and ensure extension is installed
    - CREATE TABLE with USING deeplake
    - INSERT rows, verify row count
    - Cleanup
    """
    conn = await connect_postgres()
    target_conn = None
    try:
        await conn.execute("DROP DATABASE IF EXISTS test_dl_works_db")
        await conn.execute("CREATE DATABASE test_dl_works_db")

        target_conn = await connect_database("test_dl_works_db")
        await ensure_extension(target_conn)

        await target_conn.execute("""
            CREATE TABLE test_vectors (
                id SERIAL PRIMARY KEY,
                v1 float4[]
            ) USING deeplake
        """)

        await target_conn.execute("""
            INSERT INTO test_vectors (v1) VALUES
                (ARRAY[1.0, 2.0, 3.0]),
                (ARRAY[4.0, 5.0, 6.0]),
                (ARRAY[7.0, 8.0, 9.0])
        """)

        count = await target_conn.fetchval("SELECT count(*) FROM test_vectors")
        assert count == 3, f"Expected 3 rows, got {count}"

        await target_conn.execute("DROP TABLE test_vectors")
    finally:
        if target_conn is not None:
            await target_conn.close()
        await conn.execute("DROP DATABASE IF EXISTS test_dl_works_db")
        await conn.close()


@pytest.mark.asyncio
async def test_drop_database_with_extension(pg_server):
    """
    Verify that DROP DATABASE succeeds on a database with pg_deeplake installed.

    Steps:
    - CREATE DATABASE test_drop_db
    - Connect, verify extension exists, disconnect
    - DROP DATABASE -- should succeed without errors
    """
    conn = await connect_postgres()
    try:
        await conn.execute("DROP DATABASE IF EXISTS test_drop_db")
        await conn.execute("CREATE DATABASE test_drop_db")

        target_conn = await connect_database("test_drop_db")
        await ensure_extension(target_conn)

        ext = await target_conn.fetchval(
            "SELECT extname FROM pg_extension WHERE extname = 'pg_deeplake'"
        )
        assert ext == "pg_deeplake"
        await target_conn.close()

        await conn.execute("DROP DATABASE test_drop_db")

        # Verify database no longer exists
        exists = await conn.fetchval(
            "SELECT 1 FROM pg_database WHERE datname = 'test_drop_db'"
        )
        assert exists is None, "Database should not exist after DROP"
    finally:
        await conn.execute("DROP DATABASE IF EXISTS test_drop_db")
        await conn.close()


@pytest.mark.asyncio
async def test_create_multiple_databases(pg_server):
    """
    Verify that pg_deeplake works across multiple newly created databases.

    Steps:
    - CREATE DATABASE test_multi_db_1 and test_multi_db_2
    - Verify both have the extension installed
    - Cleanup: DROP both
    """
    conn = await connect_postgres()
    conn1 = None
    conn2 = None
    try:
        await conn.execute("DROP DATABASE IF EXISTS test_multi_db_1")
        await conn.execute("DROP DATABASE IF EXISTS test_multi_db_2")
        await conn.execute("CREATE DATABASE test_multi_db_1")
        await conn.execute("CREATE DATABASE test_multi_db_2")

        conn1 = await connect_database("test_multi_db_1")
        conn2 = await connect_database("test_multi_db_2")

        await ensure_extension(conn1)
        await ensure_extension(conn2)

        ext1 = await conn1.fetchval(
            "SELECT extname FROM pg_extension WHERE extname = 'pg_deeplake'"
        )
        ext2 = await conn2.fetchval(
            "SELECT extname FROM pg_extension WHERE extname = 'pg_deeplake'"
        )

        assert ext1 == "pg_deeplake", (
            f"Expected pg_deeplake in test_multi_db_1, got: {ext1}"
        )
        assert ext2 == "pg_deeplake", (
            f"Expected pg_deeplake in test_multi_db_2, got: {ext2}"
        )
    finally:
        if conn1 is not None:
            await conn1.close()
        if conn2 is not None:
            await conn2.close()
        await conn.execute("DROP DATABASE IF EXISTS test_multi_db_1")
        await conn.execute("DROP DATABASE IF EXISTS test_multi_db_2")
        await conn.close()


# ---------------------------------------------------------------------------
# Async auto-install tests (no manual fallback — proves the sync worker path)
# ---------------------------------------------------------------------------


async def poll_for_extension(conn, timeout=10.0):
    """Poll for pg_deeplake extension. Returns True if found, False on timeout."""
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        ext = await conn.fetchval(
            "SELECT extname FROM pg_extension WHERE extname = 'pg_deeplake'"
        )
        if ext == "pg_deeplake":
            return True
        await asyncio.sleep(0.25)
    return False


@pytest.mark.asyncio
async def test_async_extension_auto_install(pg_server):
    """
    Verify the sync worker installs pg_deeplake asynchronously without manual
    intervention.

    After CREATE DATABASE, the post-hook queues the database name into shared
    memory. The background sync worker drains the queue and installs the
    extension via libpq. This test polls WITHOUT manual fallback — if the
    extension doesn't appear within the timeout, the test FAILS.
    """
    conn = await connect_postgres()
    target_conn = None
    try:
        await conn.execute("DROP DATABASE IF EXISTS test_async_install_db")
        await conn.execute("CREATE DATABASE test_async_install_db")

        target_conn = await connect_database("test_async_install_db")

        installed = await poll_for_extension(target_conn)
        assert installed, (
            "Sync worker should have auto-installed pg_deeplake within 10s"
        )
    finally:
        if target_conn is not None:
            await target_conn.close()
        await conn.execute("DROP DATABASE IF EXISTS test_async_install_db")
        await conn.close()


@pytest.mark.asyncio
async def test_async_extension_auto_install_multiple(pg_server):
    """
    Verify that the shared-memory queue handles multiple databases queued in
    rapid succession. All should have the extension installed by the sync
    worker within the timeout — no manual fallback.
    """
    conn = await connect_postgres()
    db_names = ["test_async_batch_1", "test_async_batch_2", "test_async_batch_3"]
    conns = {}
    try:
        for db in db_names:
            await conn.execute(f"DROP DATABASE IF EXISTS {db}")
        for db in db_names:
            await conn.execute(f"CREATE DATABASE {db}")

        for db in db_names:
            conns[db] = await connect_database(db)

        # Poll all databases for extension
        remaining = set(db_names)
        deadline = asyncio.get_event_loop().time() + 10.0
        while remaining and asyncio.get_event_loop().time() < deadline:
            for db in list(remaining):
                ext = await conns[db].fetchval(
                    "SELECT extname FROM pg_extension WHERE extname = 'pg_deeplake'"
                )
                if ext == "pg_deeplake":
                    remaining.discard(db)
            if remaining:
                await asyncio.sleep(0.25)

        assert not remaining, (
            f"Sync worker should have installed pg_deeplake in all databases "
            f"within 10s, still missing: {remaining}"
        )
    finally:
        for c in conns.values():
            await c.close()
        for db in db_names:
            await conn.execute(f"DROP DATABASE IF EXISTS {db}")
        await conn.close()


# ---------------------------------------------------------------------------
# Cross-instance database creation via catalog sync
# ---------------------------------------------------------------------------


def _run_cmd(cmd, check=True):
    """Run a shell command, handling root vs non-root."""
    user = os.environ.get("USER", "postgres")
    if os.geteuid() == 0:
        result = subprocess.run(
            ["su", "-", user, "-c", cmd],
            capture_output=True, text=True,
        )
    else:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}\nstderr: {result.stderr}")
    return result


@pytest.fixture
def second_pg_instance(pg_config, temp_dir_for_postgres):
    """
    Start a second PostgreSQL instance with deeplake.root_path configured in
    postgresql.conf so that its sync worker can discover databases from the
    shared catalog.
    """
    install_dir = pg_config["install"]
    pg_ctl = install_dir / "bin" / "pg_ctl"
    initdb = install_dir / "bin" / "initdb"
    user = os.environ.get("USER", "postgres")

    tmp = Path(tempfile.mkdtemp(prefix="deeplake_second_"))
    if os.geteuid() == 0:
        shutil.chown(str(tmp), user=user, group=user)
        os.chmod(tmp, 0o777)

    data_dir = tmp / "pg_data"
    log_file = tmp / "server.log"

    # initdb
    _run_cmd(f"{initdb} -D {data_dir} -U {user}")

    # Configure
    with open(data_dir / "postgresql.conf", "a") as f:
        f.write(f"\nport = {SECOND_INSTANCE_PORT}\n")
        f.write("shared_preload_libraries = 'pg_deeplake'\n")
        f.write("max_connections = 100\n")
        f.write("shared_buffers = 64MB\n")
        f.write(f"deeplake.root_path = '{temp_dir_for_postgres}'\n")

    # Start
    lib_path = str(install_dir / "lib")
    ld = f"{lib_path}:{os.environ.get('LD_LIBRARY_PATH', '')}"
    if os.geteuid() == 0:
        subprocess.run(
            ["su", "-", user, "-c",
             f"LD_LIBRARY_PATH={ld} {pg_ctl} -D {data_dir} -l {log_file} start"],
            check=True,
        )
    else:
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = ld
        subprocess.run(
            [str(pg_ctl), "-D", str(data_dir), "-l", str(log_file), "start"],
            check=True, env=env,
        )
    time.sleep(3)

    yield {
        "port": SECOND_INSTANCE_PORT,
        "root_path": temp_dir_for_postgres,
        "data_dir": data_dir,
        "log_file": log_file,
    }

    # Cleanup
    _run_cmd(f"{pg_ctl} stop -D {data_dir} -m fast", check=False)
    time.sleep(1)
    if tmp.exists():
        shutil.rmtree(tmp)


@pytest.mark.asyncio
async def test_database_creation_synced_to_second_instance(
    pg_server, second_pg_instance, temp_dir_for_postgres,
):
    """
    Verify cross-instance database creation via the shared catalog.

    Instance A (primary, port 5432):
      - SET deeplake.root_path → CREATE DATABASE → catalog records the DB.

    Instance B (secondary, port 5434):
      - Has the same deeplake.root_path in postgresql.conf.
      - Its sync worker reads the catalog, discovers the new database, creates
        it locally via CREATE DATABASE, and installs pg_deeplake.

    The test polls Instance B for the database and extension without manual
    intervention.
    """
    root_path = temp_dir_for_postgres
    port_b = second_pg_instance["port"]
    db_name = "test_cross_instance_db"

    # --- Instance A: create the database with catalog recording ---
    conn_a = await connect_postgres()
    try:
        await conn_a.execute(f"SET deeplake.root_path = '{root_path}'")
        await conn_a.execute(f"DROP DATABASE IF EXISTS {db_name}")
        await conn_a.execute(f"CREATE DATABASE {db_name}")
    finally:
        await conn_a.close()

    # --- Instance B: wait for the sync worker to create the database ---
    deadline = asyncio.get_event_loop().time() + 15.0
    db_exists = False
    while asyncio.get_event_loop().time() < deadline:
        try:
            conn_b = await connect_postgres(port=port_b)
            exists = await conn_b.fetchval(
                "SELECT 1 FROM pg_database WHERE datname = $1", db_name,
            )
            await conn_b.close()
            if exists:
                db_exists = True
                break
        except Exception:
            pass
        await asyncio.sleep(0.5)

    assert db_exists, (
        f"Sync worker on Instance B should have created '{db_name}' "
        f"within 15s via catalog sync"
    )

    # --- Instance B: verify the extension was installed in the new DB ---
    target_conn = await connect_database(db_name, port=port_b)
    try:
        installed = await poll_for_extension(target_conn, timeout=10.0)
        assert installed, (
            f"pg_deeplake should be auto-installed in '{db_name}' on Instance B"
        )
    finally:
        await target_conn.close()

    # --- Cleanup on both instances ---
    conn_a = await connect_postgres()
    try:
        await conn_a.execute(f"DROP DATABASE IF EXISTS {db_name}")
    finally:
        await conn_a.close()

    try:
        conn_b = await connect_postgres(port=port_b)
        await conn_b.execute(f"DROP DATABASE IF EXISTS {db_name}")
        await conn_b.close()
    except Exception:
        pass
