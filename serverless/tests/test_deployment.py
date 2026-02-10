#!/usr/bin/env python3
"""
Integration tests for pg_deeplake serverless deployment.

Run with: pytest tests/test_deployment.py -v

Prerequisites:
    docker compose up -d   (from serverless/ directory)
    All instances healthy

Instance count is read dynamically from config/.instance_count (default 2).

Upstream pg_deeplake bugs affecting tests (tracked for fix):
    - Sync worker SIGPIPE (exit 141): Upstream bug where the sync worker
      crashes on first table sync. Postmaster auto-restarts it and the
      second sync succeeds. Tests use extended waits + retries to work
      around this until the upstream fix lands.
    - DROP TABLE/DATABASE crash: Upstream bug where these DDL operations
      crash the extension process. Tests avoid destructive cleanup as a
      workaround. This is NOT expected behavior.
"""

import os
import subprocess
import sys
import time
import urllib.error
import uuid
from pathlib import Path

import psycopg2
import pytest


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HAPROXY_HOST = "localhost"
HAPROXY_PORT = 5432

_SERVERLESS_DIR = Path(__file__).resolve().parent.parent
_INSTANCE_COUNT_FILE = _SERVERLESS_DIR / "config" / ".instance_count"

PG_USER = "postgres"
PG_PASSWORD = "postgres"
PG_DB = "postgres"

# Optional fast mode for local iteration:
#   FAST_TESTS=1 pytest tests/test_deployment.py -v
# Tune individual timings with env vars below if needed.
FAST_TESTS = os.environ.get("FAST_TESTS", "0") == "1"


def _int_env(name, default):
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


# Extra wait for sync + potential crash recovery cycle.
SYNC_WAIT = _int_env("SYNC_WAIT_SECONDS", 6 if FAST_TESTS else 20)
PG_CONNECT_RETRIES = _int_env("PG_CONNECT_RETRIES", 3 if FAST_TESTS else 5)
PG_CONNECT_DELAY = _int_env("PG_CONNECT_DELAY_SECONDS", 1 if FAST_TESTS else 3)
QUERY_RETRIES = _int_env("QUERY_RETRIES", 2 if FAST_TESTS else 3)
QUERY_DELAY = _int_env("QUERY_DELAY_SECONDS", 2 if FAST_TESTS else 5)
INSTANCE_WAIT_TIMEOUT = _int_env("INSTANCE_WAIT_TIMEOUT_SECONDS", 30 if FAST_TESTS else 60)
FAILOVER_DETECT_WAIT = _int_env("FAILOVER_DETECT_WAIT_SECONDS", 8 if FAST_TESTS else 15)
METRICS_RETRIES = _int_env("METRICS_RETRIES", 5 if FAST_TESTS else 10)
METRICS_DELAY = _int_env("METRICS_DELAY_SECONDS", 1 if FAST_TESTS else 2)


def _read_instance_count():
    """Read instance count from config/.instance_count (default 2)."""
    try:
        return int(_INSTANCE_COUNT_FILE.read_text().strip())
    except (FileNotFoundError, ValueError):
        return 2


def get_instances():
    """Return instance list based on current count."""
    n = _read_instance_count()
    return [
        {"name": f"pg-deeplake-{i}", "host": "localhost", "port": 5432 + i}
        for i in range(1, n + 1)
    ]


INSTANCES = get_instances()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pg_connect(host, port, database=None, retries=None, delay=None):
    """Create a psycopg2 connection with retry for transient recovery states."""
    retries = PG_CONNECT_RETRIES if retries is None else retries
    delay = PG_CONNECT_DELAY if delay is None else delay
    last_err = None
    for attempt in range(retries):
        try:
            return psycopg2.connect(
                host=host,
                port=port,
                database=database or PG_DB,
                user=PG_USER,
                password=PG_PASSWORD,
                connect_timeout=10,
            )
        except psycopg2.OperationalError as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(delay)
    raise last_err


def wait_for_instance(host, port, timeout=None):
    """Wait until an instance accepts connections."""
    timeout = INSTANCE_WAIT_TIMEOUT if timeout is None else timeout
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            conn = psycopg2.connect(
                host=host, port=port, user=PG_USER, password=PG_PASSWORD,
                database=PG_DB, connect_timeout=3,
            )
            conn.close()
            return True
        except Exception:
            time.sleep(2)
    return False


def query_with_retry(host, port, sql, database=None, retries=None, delay=None):
    """Execute a query with retries (handles crash-recovery cycles)."""
    retries = QUERY_RETRIES if retries is None else retries
    delay = QUERY_DELAY if delay is None else delay
    last_err = None
    for attempt in range(retries):
        try:
            conn = pg_connect(host, port, database=database, retries=3, delay=2)
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute(sql)
                if cur.description:
                    result = cur.fetchall()
                else:
                    result = None
            conn.close()
            return result
        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(delay)
    raise last_err


def fetch_http_with_retry(url, timeout=5, retries=None, delay=None):
    """Fetch HTTP URL with retries for transient resets during restarts."""
    import urllib.request

    retries = METRICS_RETRIES if retries is None else retries
    delay = METRICS_DELAY if delay is None else delay
    last_err = None
    for attempt in range(retries):
        try:
            resp = urllib.request.urlopen(url, timeout=timeout)
            return resp
        except (urllib.error.URLError, ConnectionResetError, TimeoutError) as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(delay)
    raise last_err


# ---------------------------------------------------------------------------
# Tests — ordered so DDL-triggering tests run last
# ---------------------------------------------------------------------------

class Test01Connectivity:
    """Basic connectivity and HAProxy routing tests."""

    def test_haproxy_accepts_connections(self):
        """HAProxy should accept PostgreSQL connections on port 5432."""
        rows = query_with_retry(HAPROXY_HOST, HAPROXY_PORT, "SELECT 1")
        assert rows[0][0] == 1

    def test_direct_instance_connections(self):
        """Each backend instance should accept direct connections."""
        for inst in INSTANCES:
            rows = query_with_retry(inst["host"], inst["port"], "SELECT 1")
            assert rows[0][0] == 1, f"{inst['name']} failed SELECT 1"

    def test_haproxy_routes_to_multiple_backends(self):
        """HAProxy round-robin should distribute across backends."""
        if len(INSTANCES) < 2:
            pytest.skip("Need >= 2 instances for round-robin test")

        backend_addrs = set()
        for _ in range(10):
            rows = query_with_retry(
                HAPROXY_HOST, HAPROXY_PORT, "SELECT inet_server_addr()"
            )
            backend_addrs.add(str(rows[0][0]))

        assert len(backend_addrs) >= 2, (
            f"Expected requests routed to >=2 backends, got {backend_addrs}"
        )


class Test02StatelessMode:
    """Verify stateless mode is properly configured on all instances."""

    def test_stateless_enabled_on_all_instances(self):
        """deeplake.stateless_enabled should be 'on' on every instance."""
        for inst in INSTANCES:
            rows = query_with_retry(
                inst["host"], inst["port"], "SHOW deeplake.stateless_enabled"
            )
            val = rows[0][0]
            assert val in ("on", "true"), (
                f"{inst['name']}: stateless_enabled = {val}, expected on/true"
            )

    def test_root_path_configured(self):
        """deeplake.root_path should be set (S3 or local path) on all instances."""
        for inst in INSTANCES:
            rows = query_with_retry(
                inst["host"], inst["port"], "SHOW deeplake.root_path"
            )
            val = rows[0][0]
            assert val.startswith("s3://") or val.startswith("/"), (
                f"{inst['name']}: root_path = {val}, expected s3:// or absolute path"
            )

    def test_same_root_path_all_instances(self):
        """All instances must share the same root_path for catalog consistency."""
        paths = []
        for inst in INSTANCES:
            rows = query_with_retry(
                inst["host"], inst["port"], "SHOW deeplake.root_path"
            )
            paths.append(rows[0][0])
        assert len(set(paths)) == 1, f"Root paths differ: {paths}"

    def test_deeplake_extension_loaded(self):
        """The pg_deeplake extension should be installed on all instances."""
        for inst in INSTANCES:
            rows = query_with_retry(
                inst["host"], inst["port"],
                "SELECT extversion FROM pg_extension WHERE extname = 'pg_deeplake'",
            )
            assert rows and len(rows) > 0, (
                f"{inst['name']}: pg_deeplake extension not found"
            )

    def test_stateless_via_haproxy(self):
        """Stateless mode should be visible through HAProxy too."""
        rows = query_with_retry(
            HAPROXY_HOST, HAPROXY_PORT, "SHOW deeplake.stateless_enabled"
        )
        val = rows[0][0]
        assert val in ("on", "true")


class Test03HAProxyStats:
    """Test HAProxy stats/metrics endpoints."""

    def test_stats_page_accessible(self):
        """HAProxy stats page should be accessible when auth is configured."""
        import urllib.request

        stats_user = os.environ.get("HAPROXY_STATS_USER")
        stats_password = os.environ.get("HAPROXY_STATS_PASSWORD")
        if not (stats_user and stats_password):
            pytest.skip("Stats UI disabled unless HAPROXY_STATS_USER/PASSWORD are set")

        auth_handler = urllib.request.HTTPBasicAuthHandler()
        auth_handler.add_password(
            realm="HAProxy Statistics",
            uri="http://localhost:8404/stats",
            user=stats_user,
            passwd=stats_password,
        )
        opener = urllib.request.build_opener(auth_handler)
        resp = opener.open("http://localhost:8404/stats", timeout=5)
        assert resp.status == 200

    def test_prometheus_metrics(self):
        """Prometheus metrics endpoint should return metrics (no auth needed)."""
        resp = fetch_http_with_retry("http://localhost:8404/metrics", timeout=5)
        body = resp.read().decode()
        assert "haproxy_" in body, "Expected haproxy_ metrics prefix"


class Test04DDLPropagation:
    """Test that DDL on one instance propagates to others via S3 catalog sync.

    These tests create deeplake tables which may trigger sync worker crashes.
    They run after the simple tests to avoid disrupting them.
    """

    @pytest.mark.skipif(
        _read_instance_count() < 2,
        reason="Need >= 2 instances for DDL propagation test",
    )
    def test_create_table_propagates(self):
        """CREATE TABLE on instance 1 should become visible on instance 2."""
        table_name = f"test_prop_{uuid.uuid4().hex[:8]}"
        inst1 = INSTANCES[0]
        inst2 = INSTANCES[-1]  # last instance (works for any N)

        # Create on instance 1
        conn1 = pg_connect(inst1["host"], inst1["port"])
        conn1.autocommit = True
        with conn1.cursor() as cur:
            cur.execute(
                f"CREATE TABLE {table_name} (id INTEGER, val TEXT) USING deeplake"
            )
        conn1.close()

        # Wait for catalog sync (including potential crash-recovery cycle)
        time.sleep(SYNC_WAIT)

        # Verify on last instance (with retry for recovery)
        rows = query_with_retry(
            inst2["host"], inst2["port"],
            f"SELECT 1 FROM pg_tables WHERE tablename = '{table_name}' "
            f"AND schemaname = 'public'",
        )
        assert rows and len(rows) > 0, (
            f"Table {table_name} not visible on {inst2['name']} after sync"
        )


class Test05DataVisibility:
    """Test that data written through one path is readable through another."""

    @pytest.mark.skipif(
        _read_instance_count() < 2,
        reason="Need >= 2 instances for cross-instance data visibility",
    )
    def test_insert_via_haproxy_visible_on_backends(self):
        """INSERT through HAProxy should be queryable on each backend directly."""
        table_name = f"test_data_{uuid.uuid4().hex[:8]}"

        # Create table on instance 1
        query_with_retry(
            INSTANCES[0]["host"],
            INSTANCES[0]["port"],
            f"CREATE TABLE {table_name} (id INTEGER, name TEXT) USING deeplake",
            retries=max(QUERY_RETRIES, 3),
            delay=QUERY_DELAY,
        )

        time.sleep(SYNC_WAIT)

        # Insert through HAProxy (with retry — HAProxy may route to a
        # recovering instance)
        for attempt in range(3):
            try:
                ha_conn = pg_connect(HAPROXY_HOST, HAPROXY_PORT)
                with ha_conn.cursor() as cur:
                    cur.execute(
                        f"INSERT INTO {table_name} VALUES (1, 'test_row')"
                    )
                ha_conn.commit()
                ha_conn.close()
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(5)

        time.sleep(SYNC_WAIT)

        # Verify on each backend (with retry for crash recovery)
        for inst in INSTANCES:
            rows = query_with_retry(
                inst["host"], inst["port"],
                f"SELECT count(*) FROM {table_name}",
            )
            count = rows[0][0]
            assert count >= 1, (
                f"{inst['name']}: expected >=1 rows, got {count}"
            )


MANAGE_PY = str(_SERVERLESS_DIR / "scripts" / "manage.py")

SCALE_ZERO_TIMEOUT = _int_env("SCALE_ZERO_TIMEOUT_SECONDS", 60 if FAST_TESTS else 120)


class Test06Failover:
    """Test HAProxy failover behavior when an instance goes down.

    This test is last because stopping/starting containers can disrupt
    other tests.
    """

    @pytest.mark.skipif(
        _read_instance_count() < 2,
        reason="Need >= 2 instances for failover test",
    )
    @pytest.mark.skipif(
        os.environ.get("SKIP_FAILOVER") == "1",
        reason="Failover test disabled",
    )
    def test_haproxy_failover_on_instance_stop(self):
        """Stopping one instance should not break HAProxy connectivity."""
        # Verify all instances are reachable
        for inst in INSTANCES:
            assert wait_for_instance(inst["host"], inst["port"], timeout=30), (
                f"{inst['name']} not reachable before failover test"
            )

        # Stop the last instance
        last = INSTANCES[-1]
        subprocess.run(
            ["docker", "stop", "-t", "5", last["name"]],
            check=True,
            capture_output=True,
        )

        try:
            # HAProxy checks every 3s, needs fall=3 failures = ~9s in normal mode.
            time.sleep(FAILOVER_DETECT_WAIT)

            # HAProxy should still work via remaining instances
            conn = pg_connect(HAPROXY_HOST, HAPROXY_PORT, retries=5, delay=3)
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                assert cur.fetchone()[0] == 1
            conn.close()

        finally:
            # Always restart the stopped instance
            subprocess.run(
                ["docker", "start", last["name"]],
                capture_output=True,
            )
            assert wait_for_instance(
                last["host"], last["port"], timeout=90
            ), f"{last['name']} did not recover after restart"


class Test07ScaleToZero:
    """Test scale-to-zero and scale-back-up via manage.py."""

    @pytest.mark.skipif(
        os.environ.get("SKIP_SCALE_ZERO") == "1",
        reason="Scale-to-zero test disabled",
    )
    def test_scale_to_zero_and_back(self):
        """Scale to 0, verify HAProxy stays up, scale to 1, verify query works."""
        original_n = _read_instance_count()

        # Scale to 0
        result = subprocess.run(
            [sys.executable, MANAGE_PY, "scale", "0"],
            capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0, f"scale 0 failed: {result.stderr}"

        try:
            # HAProxy should still be listening (connections queue)
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            err = sock.connect_ex((HAPROXY_HOST, HAPROXY_PORT))
            sock.close()
            assert err == 0, "HAProxy not accepting connections after scale-to-zero"

            # Scale back to 1
            result = subprocess.run(
                [sys.executable, MANAGE_PY, "scale", "1"],
                capture_output=True, text=True, timeout=SCALE_ZERO_TIMEOUT,
            )
            assert result.returncode == 0, f"scale 1 failed: {result.stderr}"

            # Wait for instance to be ready
            assert wait_for_instance(
                "localhost", 5433, timeout=SCALE_ZERO_TIMEOUT
            ), "pg-deeplake-1 did not become healthy after scale-up"

            # Query through HAProxy should succeed
            rows = query_with_retry(HAPROXY_HOST, HAPROXY_PORT, "SELECT 1")
            assert rows[0][0] == 1
        finally:
            # Restore original instance count
            if _read_instance_count() != original_n:
                subprocess.run(
                    [sys.executable, MANAGE_PY, "scale", str(original_n)],
                    capture_output=True, text=True, timeout=SCALE_ZERO_TIMEOUT,
                )
                for inst in get_instances():
                    wait_for_instance(inst["host"], inst["port"], timeout=90)


class Test08IdleWatch:
    """Smoke test for the idle-watch auto-scaling monitor."""

    @pytest.mark.skipif(
        os.environ.get("SKIP_IDLE_WATCH") == "1",
        reason="Idle-watch test disabled",
    )
    def test_idle_watch_scales_down_and_wakes(self):
        """Start idle-watch with short timeout, verify scale-down then wake."""
        original_n = _read_instance_count()
        if original_n == 0:
            # Need at least 1 instance running to test idle scale-down
            subprocess.run(
                [sys.executable, MANAGE_PY, "scale", "1"],
                capture_output=True, text=True, timeout=120,
            )
            wait_for_instance("localhost", 5433, timeout=90)

        env = os.environ.copy()
        env["IDLE_TIMEOUT_SECONDS"] = "10"
        env["IDLE_POLL_INTERVAL"] = "2"

        # Start idle-watch in background
        proc = subprocess.Popen(
            [sys.executable, MANAGE_PY, "idle-watch"],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )

        try:
            # Wait for idle timeout + poll cycles + scale-down time
            time.sleep(25)

            # Instances should have scaled to 0
            assert _read_instance_count() == 0, (
                "Expected 0 instances after idle timeout"
            )

            # Now make a connection to trigger wake-on-connect.
            # HAProxy will queue it; idle-watch should detect and scale up.
            # We try connecting in a subprocess so it doesn't block the test.
            connect_proc = subprocess.Popen(
                ["psql", "-h", HAPROXY_HOST, "-p", str(HAPROXY_PORT),
                 "-U", PG_USER, "-d", PG_DB, "-c", "SELECT 1"],
                env={**env, "PGPASSWORD": PG_PASSWORD},
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )

            # Wait for idle-watch to detect the queued connection and scale up
            deadline = time.time() + SCALE_ZERO_TIMEOUT
            scaled_up = False
            while time.time() < deadline:
                if _read_instance_count() > 0:
                    scaled_up = True
                    break
                time.sleep(2)

            assert scaled_up, "idle-watch did not scale up after queued connection"

            # Wait for instance health, then verify query
            wait_for_instance("localhost", 5433, timeout=SCALE_ZERO_TIMEOUT)
            connect_proc.wait(timeout=SCALE_ZERO_TIMEOUT)

            rows = query_with_retry(HAPROXY_HOST, HAPROXY_PORT, "SELECT 1")
            assert rows[0][0] == 1
        finally:
            proc.terminate()
            proc.wait(timeout=10)

            # Restore original instance count
            if _read_instance_count() != original_n:
                subprocess.run(
                    [sys.executable, MANAGE_PY, "scale", str(max(original_n, 1))],
                    capture_output=True, text=True, timeout=SCALE_ZERO_TIMEOUT,
                )
                for inst in get_instances():
                    wait_for_instance(inst["host"], inst["port"], timeout=90)
