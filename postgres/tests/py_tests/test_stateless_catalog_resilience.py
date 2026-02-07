"""
Release-risk repro tests for pg_deeplake stateless mode.

These tests are part of the default suite and should stay enabled.
Run with:
    pytest postgres/tests/py_tests/test_stateless_release_risks.py
"""

from pathlib import Path
import os
import shutil

import asyncpg
import pytest


def _sql_literal(value: str) -> str:
    """Return a single-quoted SQL literal with escaped quotes."""
    return "'" + value.replace("'", "''") + "'"


@pytest.mark.asyncio
async def test_stateless_catalog_recovers_from_legacy_non_catalog_path(db_conn: asyncpg.Connection, temp_dir_for_postgres: str):
    """
    Risk #1 repro:
    A pre-existing non-catalog object at __deeplake_catalog/tables should be migrated/recovered.

    Expected release behavior:
    - SET deeplake.root_path succeeds
    - Catalog is usable after recovery
    """
    await db_conn.execute("SET deeplake.stateless_enabled = true")

    root_path = Path(temp_dir_for_postgres) / "legacy_non_catalog_root"
    poisoned_path = root_path / "__deeplake_catalog" / "tables"
    poisoned_path.parent.mkdir(parents=True, exist_ok=True)
    poisoned_path.write_text("not a deeplake catalog table", encoding="utf-8")

    # When running as root (CI), ensure postgres user can delete the file
    if os.geteuid() == 0:
        user = os.environ.get("USER", "postgres")
        for p in [root_path, root_path / "__deeplake_catalog"]:
            shutil.chown(p, user=user, group=user)
        shutil.chown(poisoned_path, user=user, group=user)

    await db_conn.execute(f"SET deeplake.root_path = {_sql_literal(str(root_path))}")

    # If recovery worked, catalog-backed table registration should still work.
    await db_conn.execute("DROP TABLE IF EXISTS stateless_legacy_recovery")
    await db_conn.execute("CREATE TABLE stateless_legacy_recovery (id INTEGER) USING deeplake")

    count = await db_conn.fetchval(
        "SELECT COUNT(*) FROM pg_deeplake_tables WHERE table_name = 'public.stateless_legacy_recovery'"
    )
    assert count == 1


@pytest.mark.asyncio
async def test_stateless_bootstrap_permission_error_keeps_backend_alive(db_conn: asyncpg.Connection, temp_dir_for_postgres: str):
    """
    Risk #2 repro:
    Catalog bootstrap failure (permission denied) must not kill backend/session.

    Expected release behavior:
    - SET deeplake.root_path fails with a PostgreSQL error
    - Same connection remains usable afterwards
    """
    await db_conn.execute("SET deeplake.stateless_enabled = true")

    readonly_root = Path(temp_dir_for_postgres) / "readonly_root"
    readonly_root.mkdir(parents=True, exist_ok=True)
    os.chmod(readonly_root, 0o555)

    try:
        with pytest.raises(asyncpg.PostgresError):
            await db_conn.execute(f"SET deeplake.root_path = {_sql_literal(str(readonly_root))}")

        # Critical assertion: backend/session is still alive.
        health = await db_conn.fetchval("SELECT 1")
        assert health == 1
    finally:
        # Allow temp fixture cleanup.
        os.chmod(readonly_root, 0o755)
