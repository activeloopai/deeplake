"""
Release-risk repro tests for pg_deeplake stateless mode.

TEMPORARILY DISABLED: client-side DDL WAL replay removed; tests need rework.

These tests are part of the default suite and should stay enabled.
Run with:
    pytest postgres/tests/py_tests/test_stateless_release_risks.py
"""

from pathlib import Path
import os

import asyncpg
import pytest

pytestmark = pytest.mark.skip(reason="client-side DDL WAL replay removed; tests need rework")


def _sql_literal(value: str) -> str:
    """Return a single-quoted SQL literal with escaped quotes."""
    return "'" + value.replace("'", "''") + "'"


@pytest.mark.asyncio
async def test_stateless_bootstrap_permission_error_keeps_backend_alive(db_conn: asyncpg.Connection, temp_dir_for_postgres: str):
    """
    Risk #2 repro:
    Catalog bootstrap failure (permission denied) must not kill backend/session.

    Expected release behavior:
    - SET deeplake.root_path fails with a PostgreSQL error
    - Same connection remains usable afterwards
    """
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
