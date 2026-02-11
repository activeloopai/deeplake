#!/usr/bin/env python3
"""Unit tests for manage.py pure functions that affect production behavior."""

import textwrap
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from manage import (
    _ddl_guardrails,
    _strip_sql_comments,
    parse_backend_sessions,
    parse_backend_queue,
    parse_frontend_session_total,
)


# ---------------------------------------------------------------------------
# DDL guardrails — wrong result = dangerous SQL hits prod or legit DDL blocked
# ---------------------------------------------------------------------------

class TestDdlGuardrails:
    @pytest.mark.parametrize("stmt", [
        "CREATE TABLE test (id INT)",
        "ALTER TABLE test ADD COLUMN name TEXT",
        "DROP TABLE test",
        "TRUNCATE TABLE test",
        "GRANT SELECT ON test TO reader",
        "REINDEX TABLE test",
        "create table test (id int)",  # case insensitive
        "CREATE TABLE test (id INT);",  # trailing semicolon
        "/* comment */ CREATE TABLE test (id INT)",  # leading comment
    ])
    def test_allowed_ddl(self, stmt):
        ok, _ = _ddl_guardrails(stmt)
        assert ok is True

    @pytest.mark.parametrize("stmt,expected_word", [
        ("SELECT * FROM test", "SELECT"),
        ("INSERT INTO test VALUES (1)", "INSERT"),
        ("UPDATE test SET id = 1", "UPDATE"),
        ("DELETE FROM test", "DELETE"),
    ])
    def test_dml_blocked(self, stmt, expected_word):
        ok, reason = _ddl_guardrails(stmt)
        assert ok is False
        assert expected_word in reason

    def test_multi_statement_blocked(self):
        ok, reason = _ddl_guardrails("CREATE TABLE a (id INT); DROP TABLE b")
        assert ok is False
        assert "multiple" in reason

    def test_empty_blocked(self):
        ok, _ = _ddl_guardrails("")
        assert ok is False

    def test_comment_only_blocked(self):
        ok, _ = _ddl_guardrails("-- just a comment")
        assert ok is False


class TestStripSqlComments:
    def test_inline_comment(self):
        assert _strip_sql_comments("SELECT 1 -- comment") == "SELECT 1"

    def test_block_comment(self):
        result = _strip_sql_comments("SELECT /* drop tables */ 1")
        assert "drop" not in result
        assert "1" in result

    def test_multiline_block(self):
        result = _strip_sql_comments("SELECT /* hello\nworld */ 1")
        assert "hello" not in result


# ---------------------------------------------------------------------------
# HAProxy CSV parsing — wrong result = bad scaling / wake-on-connect broken
# ---------------------------------------------------------------------------

SAMPLE_CSV = textwrap.dedent("""\
    # pxname,svname,qcur,qmax,scur,smax,slim,stot,bin,bout
    pg_frontend,FRONTEND,,,1,5,1000,142,8520,125340
    pg_backends,pg-deeplake-1,0,0,3,4,,42,2140,62670
    pg_backends,pg-deeplake-2,0,0,5,6,,50,3140,32670
    pg_backends,BACKEND,2,3,8,10,200,92,5280,95340
    stats,FRONTEND,,,0,2,1000,12,1234,5678
""")


class TestParseBackendSessions:
    def test_sums_server_rows_only(self):
        # Must sum individual servers (3+5=8), NOT the BACKEND summary row
        assert parse_backend_sessions(SAMPLE_CSV) == 8

    def test_zero_when_empty(self):
        assert parse_backend_sessions("# pxname,svname,scur\n") == 0


class TestParseBackendQueue:
    def test_reads_backend_qcur(self):
        assert parse_backend_queue(SAMPLE_CSV) == 2

    def test_zero_when_empty(self):
        assert parse_backend_queue("# pxname,svname,qcur\n") == 0


class TestParseFrontendSessionTotal:
    def test_reads_pg_frontend_not_stats(self):
        # Must read pg_frontend (142), NOT stats frontend (12)
        assert parse_frontend_session_total(SAMPLE_CSV) == 142

    def test_zero_when_empty(self):
        assert parse_frontend_session_total("# pxname,svname,stot\n") == 0
