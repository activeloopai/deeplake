"""
pytest configuration and fixtures for PostgreSQL extension tests.
"""
import sys
import asyncio
import asyncpg
import pytest
import subprocess
import time
import os
import shutil
from pathlib import Path
from typing import Dict, AsyncGenerator

# Ensure the test directory is in Python path for lib imports
sys.path.insert(0, str(Path(__file__).parent))


def get_pg_config() -> Dict[str, Path]:
    """Get PostgreSQL configuration paths."""
    major_version = 18
    minor_version = 0
    # conftest.py is in postgres/tests/py_tests/
    # We need to go up to postgres/tests/ and then to cpp/
    script_path = Path(__file__).parent  # py_tests/
    tests_dir = script_path.parent  # tests/
    postgres_dir = tests_dir.parent  # postgres/
    indra_root = postgres_dir.parent  # indra/

    postgres_source = indra_root / "cpp" / ".ext" / f"postgres-REL_{major_version}_{minor_version}"
    postgres_source = postgres_source.resolve()

    extension_path = postgres_dir  # postgres/ directory contains the .so and .sql files

    return {
        "major_version": major_version,
        "source": postgres_source,
        "install": postgres_source / "install",
        "data": postgres_source / "data",
        "extension_path": extension_path,
        "log_file": tests_dir / "logs" / "pytest_server.log",
    }


def install_extension(config: Dict[str, Path]) -> None:
    """Install PostgreSQL extension files."""
    ext_dir = config["install"] / "share" / "extension"
    lib_dir = config["install"] / "lib"
    ext_path = config["extension_path"]

    # Determine dynamic library suffix
    import platform
    if platform.system() == "Darwin":
        lib_suffix = ".dylib"
    else:
        lib_suffix = ".so"

    ext_dir.mkdir(parents=True, exist_ok=True)

    # Copy extension files
    for control_file in ext_path.glob("*.control"):
        shutil.copy(control_file, ext_dir)

    for sql_file in ext_path.glob("*.sql"):
        if sql_file.name != "utils.psql":  # Skip utils
            shutil.copy(sql_file, ext_dir)

    # Copy shared library
    lib_file = ext_path / f"pg_deeplake_{config['major_version']}{lib_suffix}"
    if lib_file.exists():
        shutil.copy(lib_file, lib_dir / f"pg_deeplake{lib_suffix}")
    else:
        raise FileNotFoundError(f"Extension library not found: {lib_file}")


def is_postgres_running(pg_ctl: Path, data_dir: Path) -> bool:
    """Check if PostgreSQL server is running."""
    result = subprocess.run(
        [str(pg_ctl), "status", "-D", str(data_dir)],
        capture_output=True
    )
    return result.returncode == 0


def stop_postgres(pg_ctl: Path, data_dir: Path) -> None:
    """Stop PostgreSQL server."""
    if is_postgres_running(pg_ctl, data_dir):
        subprocess.run(
            [str(pg_ctl), "stop", "-D", str(data_dir), "-m", "fast"],
            capture_output=True
        )
        time.sleep(2)


@pytest.fixture(scope="session")
def pg_config():
    """PostgreSQL configuration for test session."""
    return get_pg_config()


@pytest.fixture(scope="session")
def pg_server(pg_config):
    """
    Start PostgreSQL server for entire test session.

    This fixture:
    - Stops any existing server
    - Initializes a fresh database cluster
    - Installs the pg_deeplake extension
    - Starts the PostgreSQL server
    - Yields control to tests
    - Stops the server on cleanup
    """
    install_dir = pg_config["install"]
    data_dir = pg_config["data"]
    log_file = pg_config["log_file"]
    pg_ctl = install_dir / "bin" / "pg_ctl"
    initdb = install_dir / "bin" / "initdb"

    # Ensure log directory exists
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Stop any existing server
    stop_postgres(pg_ctl, data_dir)

    # Remove existing data directory
    if data_dir.exists():
        shutil.rmtree(data_dir)

    # Install extension
    print(f"Installing pg_deeplake extension...")
    install_extension(pg_config)

    # Initialize database cluster
    user = os.environ.get("USER", "postgres")
    print(f"Initializing database cluster as user: {user}")
    subprocess.run(
        [str(initdb), "-D", str(data_dir), "-U", user],
        check=True,
        capture_output=True
    )

    # Configure shared_preload_libraries and increase connection limits for stress tests
    with open(data_dir / "postgresql.conf", "a") as f:
        f.write("\nshared_preload_libraries = 'pg_deeplake'\n")
        f.write("max_connections = 300\n")  # Increased for concurrent stress tests
        f.write("shared_buffers = 128MB\n")  # Increased for better performance with many connections

    # Set library path
    env = os.environ.copy()
    lib_path = str(install_dir / "lib")
    if "LD_LIBRARY_PATH" in env:
        env["LD_LIBRARY_PATH"] = f"{lib_path}:{env['LD_LIBRARY_PATH']}"
    else:
        env["LD_LIBRARY_PATH"] = lib_path

    # Start PostgreSQL server
    print(f"Starting PostgreSQL server...")
    subprocess.run(
        [str(pg_ctl), "-D", str(data_dir), "-l", str(log_file), "start"],
        check=True,
        env=env
    )
    time.sleep(3)

    # Verify server is running
    if not is_postgres_running(pg_ctl, data_dir):
        raise RuntimeError("Failed to start PostgreSQL server")

    print("PostgreSQL server started successfully")

    yield pg_config

    # Cleanup: stop server
    print("Stopping PostgreSQL server...")
    stop_postgres(pg_ctl, data_dir)


@pytest.fixture
async def db_conn(pg_server) -> AsyncGenerator[asyncpg.Connection, None]:
    """
    Create a fresh database connection for each test.

    This fixture:
    - Creates a new connection
    - Drops and recreates the pg_deeplake extension (clean slate)
    - Loads utility functions from utils.psql
    - Yields the connection to the test
    - Closes the connection on cleanup
    """
    user = os.environ.get("USER", "postgres")
    conn = await asyncpg.connect(
        database="postgres",
        user=user,
        host="localhost",
        statement_cache_size=0  # Disable statement cache to avoid issues with executor setting changes
    )

    try:
        # Setup: Clean extension state
        await conn.execute("DROP EXTENSION IF EXISTS pg_deeplake CASCADE")
        await conn.execute("CREATE EXTENSION pg_deeplake")

        # Load utility functions from utils.psql
        script_path = Path(__file__).parent  # py_tests/
        tests_dir = script_path.parent  # tests/
        utils_file = tests_dir / "sql" / "utils.psql"

        if utils_file.exists():
            with open(utils_file, 'r') as f:
                utils_sql = f.read()
                # Filter out psql meta-commands (lines starting with \)
                sql_lines = []
                for line in utils_sql.split('\n'):
                    stripped = line.strip()
                    # Skip psql meta-commands and empty lines
                    if not stripped.startswith('\\') and stripped:
                        sql_lines.append(line)

                filtered_sql = '\n'.join(sql_lines)
                # Execute the utility functions
                if filtered_sql.strip():
                    await conn.execute(filtered_sql)

        yield conn
    finally:
        # Teardown: close connection
        await conn.close()


@pytest.fixture
async def db_conn_no_extension(pg_server) -> AsyncGenerator[asyncpg.Connection, None]:
    """
    Create a database connection without loading the extension.

    Useful for tests that need to control extension lifecycle.
    """
    user = os.environ.get("USER", "postgres")
    conn = await asyncpg.connect(
        database="postgres",
        user=user,
        host="localhost",
        statement_cache_size=0  # Disable statement cache to avoid issues with executor setting changes
    )

    try:
        yield conn
    finally:
        await conn.close()


@pytest.fixture
def pg_install_dir(pg_config) -> Path:
    """Get PostgreSQL installation directory."""
    return pg_config["install"]


@pytest.fixture
def pg_data_dir(pg_config) -> Path:
    """Get PostgreSQL data directory."""
    return pg_config["data"]
