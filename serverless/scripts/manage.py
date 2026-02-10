#!/usr/bin/env python3
"""
manage.py — Control plane CLI for pg_deeplake serverless deployment.

Commands:
    status        Show all instances, health, databases, connections, GUC values
    health        Detailed health check per instance
    provision-db  CREATE DATABASE on all instances
    run-ddl       Execute DDL on instance 1 only
    rotate-creds  Update AWS creds via ALTER SYSTEM on all instances
    scale         Scale to N pg-deeplake instances (0 = scale-to-zero)
    idle-watch    Monitor HAProxy; auto scale-to-zero on idle, wake on connect
    cache-stats   Show Redis cache statistics
    cache-flush   Flush Redis cache

Usage:
    python manage.py status
    python manage.py provision-db tpch
    python manage.py run-ddl "CREATE TABLE test (id INT) USING deeplake"
    python manage.py rotate-creds
    python manage.py health
    python manage.py scale 4
    python manage.py scale 0
    python manage.py idle-watch
    python manage.py cache-stats
    python manage.py cache-flush
"""

import argparse
import csv
import io
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
import urllib.error

import psycopg2
from psycopg2 import sql


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SERVERLESS_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = SERVERLESS_DIR / "config"
INSTANCE_COUNT_FILE = CONFIG_DIR / ".instance_count"
COMPOSE_FILE = SERVERLESS_DIR / "docker-compose.yml"
HAPROXY_CFG = CONFIG_DIR / "haproxy.cfg"
SUPAVISOR_CFG = CONFIG_DIR / "supavisor.toml"

DEFAULT_IMAGE = "quay.io/activeloopai/pg-deeplake:18"
REDIS_KEY_PREFIX = "pgdl:"


def _load_dotenv():
    """Load .env file from serverless/ directory if it exists."""
    env_file = SERVERLESS_DIR / ".env"
    if not env_file.exists():
        return
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            if key and key not in os.environ:
                os.environ[key] = value


_load_dotenv()


# ---------------------------------------------------------------------------
# Instance registry
# ---------------------------------------------------------------------------

HAPROXY = {"host": "localhost", "port": 5432}

DEFAULT_USER = os.environ.get("POSTGRES_USER", "postgres")
DEFAULT_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "")
DEFAULT_DB = os.environ.get("POSTGRES_DB", "postgres")


def _require_password():
    """Fail early if no password is configured."""
    if not DEFAULT_PASSWORD:
        print("ERROR: POSTGRES_PASSWORD is not set.")
        print("  Set it in .env or export POSTGRES_PASSWORD=<password>")
        sys.exit(1)


def _read_instance_count():
    """Read instance count from config/.instance_count (default 2)."""
    try:
        return int(INSTANCE_COUNT_FILE.read_text().strip())
    except (FileNotFoundError, ValueError):
        return 2


def get_instances():
    """Return instance list based on current instance count."""
    n = _read_instance_count()
    return [
        {"name": f"pg-deeplake-{i}", "host": "localhost", "port": 5432 + i}
        for i in range(1, n + 1)
    ]


def connect(host, port, database=None, user=None, password=None, timeout=5):
    """Connect to a PostgreSQL instance."""
    return psycopg2.connect(
        host=host,
        port=port,
        database=database or DEFAULT_DB,
        user=user or DEFAULT_USER,
        password=password or DEFAULT_PASSWORD,
        connect_timeout=timeout,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def query_one(conn, sql_str):
    """Execute query and return single value."""
    with conn.cursor() as cur:
        cur.execute(sql_str)
        row = cur.fetchone()
        return row[0] if row else None


def query_all(conn, sql_str):
    """Execute query and return all rows."""
    with conn.cursor() as cur:
        cur.execute(sql_str)
        if cur.description is None:
            return []
        return cur.fetchall()


def query_col_names(conn, sql_str):
    """Execute query and return (column_names, rows)."""
    with conn.cursor() as cur:
        cur.execute(sql_str)
        if cur.description is None:
            return [], []
        cols = [d[0] for d in cur.description]
        return cols, cur.fetchall()


def print_table(headers, rows):
    """Print a formatted ASCII table."""
    if not rows:
        print("  (no data)")
        return
    widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            widths[i] = max(widths[i], len(str(val)))

    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    print(fmt.format(*["-" * w for w in widths]))
    for row in rows:
        print(fmt.format(*[str(v) for v in row]))


def _redis_connection():
    """Return a Redis connection (lazy import)."""
    import redis
    return redis.Redis(host="localhost", port=6379, decode_responses=True)


def _flush_redis_prefix(r, prefix=REDIS_KEY_PREFIX):
    """Delete all Redis keys matching the deployment prefix (not FLUSHALL)."""
    cursor = 0
    deleted = 0
    while True:
        cursor, keys = r.scan(cursor=cursor, match=f"{prefix}*", count=500)
        if keys:
            r.delete(*keys)
            deleted += len(keys)
        if cursor == 0:
            break
    return deleted


# ---------------------------------------------------------------------------
# Compose / HAProxy / Supavisor generation
# ---------------------------------------------------------------------------

def generate_compose(n, profile="haproxy"):
    """Generate docker-compose.yml for N pg-deeplake instances.

    Uses pure string generation (no PyYAML dependency).
    Image is set via $PG_DEEPLAKE_IMAGE env var or defaults to quay.io production.
    """
    image = os.environ.get("PG_DEEPLAKE_IMAGE", DEFAULT_IMAGE)
    root_path = os.environ.get("DEEPLAKE_ROOT_PATH", "")
    local_storage = root_path and not root_path.startswith("s3://")
    lines = ["services:"]

    # pg-deeplake instances
    for i in range(1, n + 1):
        lines.append(f"  pg-deeplake-{i}:")
        lines.append(f"    image: {image}")
        lines.append(f"    container_name: pg-deeplake-{i}")
        lines.append("    init: true")
        lines.append("    restart: unless-stopped")
        lines.append("    shm_size: 1g")
        lines.append("    stop_grace_period: 30s")
        lines.append("    mem_limit: ${PG_DEEPLAKE_MEM_LIMIT:-8g}")
        lines.append("    cpus: ${PG_DEEPLAKE_CPUS:-4}")
        lines.append("    environment:")
        lines.append("      POSTGRES_USER: ${POSTGRES_USER:-postgres}")
        lines.append("      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:?POSTGRES_PASSWORD must be set}")
        lines.append("      POSTGRES_HOST_AUTH_METHOD: scram-sha-256")
        lines.append("      POSTGRES_DB: ${POSTGRES_DB:-postgres}")
        lines.append("      POSTGRES_INITDB_ARGS: --auth-host=scram-sha-256")
        lines.append("      AWS_ACCESS_KEY_ID: ${AWS_ACCESS_KEY_ID:-}")
        lines.append("      AWS_SECRET_ACCESS_KEY: ${AWS_SECRET_ACCESS_KEY:-}")
        lines.append("      AWS_SESSION_TOKEN: ${AWS_SESSION_TOKEN:-}")
        if local_storage:
            lines.append("      DEEPLAKE_ROOT_PATH: /deeplake-data")
        else:
            lines.append("      DEEPLAKE_ROOT_PATH: ${DEEPLAKE_ROOT_PATH}")
        lines.append("      DEEPLAKE_SYNC_INTERVAL_MS: ${DEEPLAKE_SYNC_INTERVAL_MS:-1000}")
        lines.append("      PG_DEEPLAKE_MEMORY_LIMIT_MB: ${PG_DEEPLAKE_MEMORY_LIMIT_MB:-0}")
        lines.append("    entrypoint: ['/usr/local/bin/deeplake-entrypoint.sh']")
        lines.append("    command: ['postgres']")
        lines.append("    volumes:")
        lines.append("      - ./scripts/deeplake-entrypoint.sh:/usr/local/bin/deeplake-entrypoint.sh:ro")
        lines.append("      - ./scripts/init-deeplake-stateless.sh:/docker-entrypoint-initdb.d/3-stateless-init.sh:ro")
        lines.append("      - ./config/postgresql-overrides.conf:/etc/postgresql-overrides.conf:ro")
        lines.append("      - ./scripts/health-check.sh:/usr/local/bin/health-check.sh:ro")
        if local_storage:
            lines.append(f"      - {root_path}:/deeplake-data")
        lines.append("    ports:")
        lines.append(f'      - "{5432 + i}:5432"')
        lines.append("    healthcheck:")
        lines.append('      test: ["CMD-SHELL", "bash /usr/local/bin/health-check.sh"]')
        lines.append("      interval: 5s")
        lines.append("      timeout: 5s")
        lines.append("      retries: 30")
        lines.append("      start_period: 10s")
        lines.append("    logging:")
        lines.append("      driver: json-file")
        lines.append("      options:")
        lines.append('        max-size: "50m"')
        lines.append('        max-file: "3"')
        lines.append("    networks:")
        lines.append("      - deeplake-net")
        lines.append("")

    # HAProxy
    lines.append("  haproxy:")
    lines.append("    image: haproxy:2.9")
    lines.append("    container_name: haproxy")
    lines.append("    restart: unless-stopped")
    lines.append('    profiles: ["haproxy"]')
    lines.append("    volumes:")
    lines.append("      - ./config/haproxy.cfg:/usr/local/etc/haproxy/haproxy.cfg:ro")
    lines.append("      - ./config/certs:/etc/haproxy/certs:ro")
    lines.append("    ports:")
    lines.append('      - "5432:5432"')
    lines.append('      - "127.0.0.1:8404:8404"')
    if n > 0:
        lines.append("    depends_on:")
        for i in range(1, n + 1):
            lines.append(f"      pg-deeplake-{i}:")
            lines.append("        condition: service_healthy")
    lines.append("    logging:")
    lines.append("      driver: json-file")
    lines.append("      options:")
    lines.append('        max-size: "10m"')
    lines.append('        max-file: "3"')
    lines.append("    networks:")
    lines.append("      - deeplake-net")
    lines.append("")

    # Supavisor
    lines.append("  supavisor:")
    lines.append("    image: supabase/supavisor:2.0")
    lines.append("    container_name: supavisor")
    lines.append("    restart: unless-stopped")
    lines.append('    profiles: ["supavisor"]')
    lines.append("    ports:")
    lines.append('      - "5432:5432"')
    lines.append('      - "4000:4000"')
    lines.append("    volumes:")
    lines.append("      - ./config/supavisor.toml:/etc/supavisor/config.toml:ro")
    if n > 0:
        lines.append("    depends_on:")
        for i in range(1, n + 1):
            lines.append(f"      pg-deeplake-{i}:")
            lines.append("        condition: service_healthy")
    lines.append("    logging:")
    lines.append("      driver: json-file")
    lines.append("      options:")
    lines.append('        max-size: "10m"')
    lines.append('        max-file: "3"')
    lines.append("    networks:")
    lines.append("      - deeplake-net")
    lines.append("")

    # Redis
    lines.append("  redis:")
    lines.append("    image: redis:7-alpine")
    lines.append("    container_name: redis")
    lines.append("    restart: unless-stopped")
    lines.append("    command: >")
    lines.append("      redis-server")
    lines.append("      --maxmemory 512mb")
    lines.append("      --maxmemory-policy allkeys-lru")
    lines.append('      --save ""')
    lines.append("      --appendonly no")
    lines.append("    ports:")
    lines.append('      - "6379:6379"')
    lines.append("    healthcheck:")
    lines.append('      test: ["CMD", "redis-cli", "ping"]')
    lines.append("      interval: 5s")
    lines.append("      timeout: 3s")
    lines.append("      retries: 5")
    lines.append("    logging:")
    lines.append("      driver: json-file")
    lines.append("      options:")
    lines.append('        max-size: "10m"')
    lines.append('        max-file: "3"')
    lines.append("    networks:")
    lines.append("      - deeplake-net")
    lines.append("")

    # Network
    lines.append("networks:")
    lines.append("  deeplake-net:")
    lines.append("    driver: bridge")
    lines.append("")

    return "\n".join(lines)


def generate_haproxy_cfg(n):
    """Generate haproxy.cfg for N backend instances.

    When n=0, generates a valid config with an empty backend section.
    HAProxy will queue incoming connections (up to timeout queue) until
    backends are added back via scale-up.

    TLS: If config/certs/server.pem exists, the frontend binds with SSL.
    Place a combined cert+key PEM at config/certs/server.pem to enable.
    """
    server_lines = "\n".join(
        f"    server pg-deeplake-{i} pg-deeplake-{i}:5432 check inter 2s fall 2 rise 1"
        for i in range(1, n + 1)
    )

    # TLS bind if cert exists
    cert_path = CONFIG_DIR / "certs" / "server.pem"
    if cert_path.exists():
        frontend_bind = "    bind *:5432 ssl crt /etc/haproxy/certs/server.pem alpn h2,http/1.1"
    else:
        frontend_bind = "    bind *:5432"

    stats_user = os.environ.get("HAPROXY_STATS_USER", "")
    stats_password = os.environ.get("HAPROXY_STATS_PASSWORD", "")
    stats_ui_block = ""
    if stats_user and stats_password:
        stats_ui_block = (
            f"    stats auth {stats_user}:{stats_password}\n"
            "    stats admin if TRUE\n"
        )

    return f"""\
global
    log stdout format raw local0 info
    maxconn 1000

defaults
    log     global
    mode    tcp
    option  tcplog
    option  dontlognull

    timeout connect 10s
    timeout client  1h
    timeout server  1h
    timeout queue   120s

    retries 3
    option redispatch

frontend pg_frontend
{frontend_bind}
    default_backend pg_backends

backend pg_backends
    balance roundrobin
    option pgsql-check user postgres

{server_lines}

frontend stats
    bind *:8404
    mode http
    http-request use-service prometheus-exporter if {{ path /metrics }}
    stats enable
    stats uri /stats
    stats refresh 10s
{stats_ui_block}"""


def generate_supavisor_cfg(n):
    """Generate supavisor.toml for N backend instances."""
    upstream_entries = ",\n".join(
        f'  {{ host = "pg-deeplake-{i}", port = 5432 }}'
        for i in range(1, n + 1)
    )

    return f"""\
# Supavisor configuration for pg_deeplake serverless
# Generated by manage.py — do not edit manually

[proxy]
listen_port = 5432
admin_port = 4000
mode = "session"

[proxy.pool]
size = 20
max_overflow = 10

[auth]
user = "postgres"
password = "postgres"
database = "postgres"

[[upstream]]
servers = [
{upstream_entries}
]
strategy = "round_robin"
health_check_interval = "3s"
"""


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_status(args):
    """Show status of all instances and HAProxy."""
    instances = get_instances()
    headers = ["Instance", "Endpoint", "Status", "Stateless", "Root Path", "Conns", "Databases"]
    num_cols = len(headers)

    # Per-instance status
    print("=== Instance Status ===\n")
    rows = []
    for inst in instances:
        try:
            conn = connect(inst["host"], inst["port"])
            conn.autocommit = True

            stateless = query_one(conn, "SHOW deeplake.stateless_enabled") or "?"
            root_path = query_one(conn, "SHOW deeplake.root_path") or "?"
            num_conns = query_one(conn, """
                SELECT count(*) FROM pg_stat_activity
                WHERE state IS NOT NULL AND pid != pg_backend_pid()
            """)
            dbs = query_all(conn, """
                SELECT datname FROM pg_database
                WHERE datistemplate = false ORDER BY datname
            """)
            conn.close()

            row = [
                inst["name"],
                f"{inst['host']}:{inst['port']}",
                "UP",
                stateless,
                root_path,
                str(num_conns),
                ", ".join(d[0] for d in dbs),
            ]
        except Exception as e:
            row = [
                inst["name"],
                f"{inst['host']}:{inst['port']}",
                "DOWN",
                str(e)[:60],
                "", "", "",
            ]

        while len(row) < num_cols:
            row.append("")
        rows.append(row[:num_cols])

    print_table(headers, rows)

    # HAProxy check
    print("\n=== Load Balancer ===\n")
    try:
        conn = connect(HAPROXY["host"], HAPROXY["port"])
        conn.autocommit = True
        backend = query_one(conn, "SELECT inet_server_port()")
        print(f"  LB {HAPROXY['host']}:{HAPROXY['port']} -> backend port {backend}  [OK]")
        conn.close()
    except Exception as e:
        print(f"  LB {HAPROXY['host']}:{HAPROXY['port']}  [FAILED: {e}]")

    print()


def cmd_health(args):
    """Detailed health check per instance."""
    instances = get_instances()

    gucs = [
        "deeplake.stateless_enabled",
        "deeplake.root_path",
        "deeplake.sync_interval_ms",
        "pg_deeplake.memory_limit_mb",
    ]

    for inst in instances:
        print(f"--- {inst['name']} ({inst['host']}:{inst['port']}) ---")
        try:
            conn = connect(inst["host"], inst["port"])
            conn.autocommit = True

            ver = query_one(conn, "SELECT version()")
            print(f"  Version: {ver}")

            uptime = query_one(conn, """
                SELECT now() - pg_postmaster_start_time()
            """)
            print(f"  Uptime: {uptime}")

            print("  GUCs:")
            for guc in gucs:
                try:
                    val = query_one(conn, f"SHOW {guc}")
                    print(f"    {guc} = {val}")
                except Exception:
                    print(f"    {guc} = (not available)")

            ext = query_one(conn, """
                SELECT extversion FROM pg_extension WHERE extname = 'pg_deeplake'
            """)
            print(f"  Extension: deeplake {ext or '(not found)'}")

            cols, conns = query_col_names(conn, """
                SELECT datname, usename, state, count(*)
                FROM pg_stat_activity
                WHERE pid != pg_backend_pid()
                GROUP BY datname, usename, state
                ORDER BY datname, state
            """)
            if conns:
                print("  Connections:")
                for r in conns:
                    print(f"    {r[0]}/{r[1]}: {r[2]} ({r[3]})")

            tables = query_all(conn, """
                SELECT schemaname, tablename
                FROM pg_tables
                WHERE schemaname = 'public'
                ORDER BY tablename
            """)
            if tables:
                print(f"  Tables ({len(tables)}):")
                for t in tables:
                    print(f"    {t[0]}.{t[1]}")

            conn.close()
            print("  Status: HEALTHY\n")

        except Exception as e:
            print(f"  Status: UNHEALTHY - {e}\n")


def cmd_provision_db(args):
    """Create a database on all instances."""
    db_name = args.name
    instances = get_instances()

    print(f"Provisioning database '{db_name}' on all instances...")

    for inst in instances:
        try:
            conn = connect(inst["host"], inst["port"])
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM pg_database WHERE datname = %s", (db_name,)
                )
                if cur.fetchone():
                    print(f"  {inst['name']}: already exists")
                    conn.close()
                    continue

                cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(db_name)))
            conn.close()
            print(f"  {inst['name']}: created")
        except Exception as e:
            print(f"  {inst['name']}: FAILED - {e}")
            sys.exit(1)

    print(f"\nDatabase '{db_name}' provisioned on all {len(instances)} instances.")


def cmd_run_ddl(args):
    """Execute DDL on instance 1 only. Catalog sync propagates to others."""
    ddl = args.sql
    database = args.database or DEFAULT_DB
    instances = get_instances()
    inst = instances[0]

    print(f"Executing DDL on {inst['name']} (database: {database}):")
    print(f"  {ddl}")
    print()

    try:
        conn = connect(inst["host"], inst["port"], database=database)
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(ddl)
        conn.close()
        print("  DDL executed successfully.")

        if len(instances) > 1:
            sync_ms = int(os.environ.get("DEEPLAKE_SYNC_INTERVAL_MS", "1000"))
            wait = (sync_ms / 1000) + 1
            print(f"  Waiting {wait:.0f}s for catalog sync to propagate...")
            time.sleep(wait)
            print("  Sync window elapsed. Verify with: manage.py health")

        # Flush deployment-prefixed Redis keys after DDL (schema may have changed)
        _try_flush_redis("DDL execution")

    except Exception as e:
        print(f"  ERROR: {e}")
        sys.exit(1)


def cmd_rotate_creds(args):
    """Rotate AWS credentials on all instances.

    Writes new creds to /dev/shm/deeplake-creds.conf (tmpfs) inside each
    container via docker exec, then reloads PostgreSQL config. Credentials
    are never written to persistent storage.
    """
    aws_key = args.aws_access_key_id or os.environ.get("AWS_ACCESS_KEY_ID")
    aws_secret = args.aws_secret_access_key or os.environ.get("AWS_SECRET_ACCESS_KEY")
    aws_token = args.aws_session_token or os.environ.get("AWS_SESSION_TOKEN", "")

    if not aws_key or not aws_secret:
        print("ERROR: AWS credentials required. Provide via args or environment.")
        print("  --aws-access-key-id / $AWS_ACCESS_KEY_ID")
        print("  --aws-secret-access-key / $AWS_SECRET_ACCESS_KEY")
        sys.exit(1)

    creds = {
        "aws_access_key_id": aws_key,
        "aws_secret_access_key": aws_secret,
    }
    if aws_token:
        creds["session_token"] = aws_token

    creds_json = json.dumps(creds)
    # Escape single quotes for PostgreSQL GUC
    creds_escaped = creds_json.replace("'", "''")

    instances = get_instances()

    print("Rotating credentials on all instances (tmpfs)...")
    failures = []
    for inst in instances:
        container = inst["name"]
        # Write creds to tmpfs inside the container
        conf_content = f"deeplake.creds = '{creds_escaped}'"
        result = subprocess.run(
            ["docker", "exec", container, "sh", "-c",
             f"echo \"{conf_content}\" > /dev/shm/deeplake-creds.conf && chmod 600 /dev/shm/deeplake-creds.conf"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"  {container}: FAILED to write tmpfs - {result.stderr.strip()}")
            failures.append(container)
            continue

        # Reload PostgreSQL to pick up new creds
        try:
            conn = connect(inst["host"], inst["port"])
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute("SELECT pg_reload_conf()")
            conn.close()
            print(f"  {container}: OK")
        except Exception as e:
            print(f"  {container}: FAILED reload - {e}")
            failures.append(container)

    # Flush deployment-prefixed Redis keys after cred rotation
    _try_flush_redis("credential rotation")

    if failures:
        print(f"\nERROR: Credential rotation failed on: {', '.join(failures)}")
        print("WARNING: Instances may have inconsistent credentials.")
        sys.exit(1)

    print("\nCredentials rotated (tmpfs). No credentials on persistent storage.")


def get_instance_count():
    """Return the current number of pg-deeplake instances."""
    return _read_instance_count()


def cmd_scale(args):
    """Scale to N pg-deeplake instances (0 = scale-to-zero, keeps HAProxy + Redis)."""
    n = args.count
    if n < 0:
        print("ERROR: Instance count must be >= 0")
        sys.exit(1)

    if args.image:
        os.environ["PG_DEEPLAKE_IMAGE"] = args.image

    image = os.environ.get("PG_DEEPLAKE_IMAGE", DEFAULT_IMAGE)
    old_n = _read_instance_count()
    print(f"Scaling: {old_n} -> {n} instances (image: {image})")

    # 1. Write instance count
    INSTANCE_COUNT_FILE.write_text(f"{n}\n")

    # 2. Generate configs
    # When scaling to 0, keep the HAProxy config with server entries from
    # the previous count so HAProxy can queue connections to the (down)
    # backends. This enables wake-on-connect: connections queue instead of
    # getting an immediate 503.
    compose_content = generate_compose(n)
    COMPOSE_FILE.write_text(compose_content)
    print(f"  Generated {COMPOSE_FILE}")

    haproxy_server_count = max(n, old_n) if n == 0 else n
    haproxy_content = generate_haproxy_cfg(haproxy_server_count)
    HAPROXY_CFG.write_text(haproxy_content)
    print(f"  Generated {HAPROXY_CFG}")

    supavisor_content = generate_supavisor_cfg(n)
    SUPAVISOR_CFG.write_text(supavisor_content)
    print(f"  Generated {SUPAVISOR_CFG}")

    # 3. docker compose up
    print("\n  Running docker compose up -d --remove-orphans ...")
    result = subprocess.run(
        ["docker", "compose", "up", "-d", "--remove-orphans"],
        cwd=str(SERVERLESS_DIR),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  ERROR: docker compose failed:\n{result.stderr}")
        sys.exit(1)
    print("  Docker compose updated.")

    # 3b. Reload HAProxy to pick up config changes (mounted volume)
    subprocess.run(
        ["docker", "kill", "-s", "HUP", "haproxy"],
        capture_output=True, text=True,
    )

    # 4. Wait for new instances to be healthy
    if n > old_n:
        print(f"\n  Waiting for new instances ({old_n + 1}..{n}) to become healthy...")
        for i in range(old_n + 1, n + 1):
            name = f"pg-deeplake-{i}"
            print(f"    Waiting for {name}...", end="", flush=True)
            deadline = time.time() + 120
            healthy = False
            while time.time() < deadline:
                result = subprocess.run(
                    ["docker", "inspect", "--format", "{{.State.Health.Status}}", name],
                    capture_output=True, text=True,
                )
                if result.stdout.strip() == "healthy":
                    healthy = True
                    break
                time.sleep(1)
            if healthy:
                print(" OK")
            else:
                print(" TIMEOUT (may still be starting)")

    # 5. Provision existing databases on new instances
    if n > old_n:
        _provision_new_instances(old_n, n)

    print(f"\nScale complete: {n} instance(s) running.")


def _provision_new_instances(old_n, new_n):
    """Provision existing databases on newly added instances."""
    # Get databases from instance 1
    try:
        conn = connect("localhost", 5433)
        conn.autocommit = True
        dbs = query_all(conn, """
            SELECT datname FROM pg_database
            WHERE datistemplate = false AND datname != 'postgres'
            ORDER BY datname
        """)
        conn.close()
    except Exception:
        return

    if not dbs:
        return

    db_names = [d[0] for d in dbs]
    print(f"\n  Provisioning databases {db_names} on new instances...")

    for i in range(old_n + 1, new_n + 1):
        port = 5432 + i
        name = f"pg-deeplake-{i}"
        for db_name in db_names:
            try:
                conn = connect("localhost", port)
                conn.autocommit = True
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT 1 FROM pg_database WHERE datname = %s", (db_name,)
                    )
                    if cur.fetchone():
                        print(f"    {name}/{db_name}: already exists")
                        conn.close()
                        continue
                    cur.execute(
                        sql.SQL("CREATE DATABASE {}").format(sql.Identifier(db_name))
                    )
                conn.close()
                print(f"    {name}/{db_name}: created")
            except Exception as e:
                print(f"    {name}/{db_name}: FAILED - {e}")


def _try_flush_redis(reason):
    """Flush deployment-prefixed Redis keys. Does not use FLUSHALL."""
    try:
        r = _redis_connection()
        deleted = _flush_redis_prefix(r, REDIS_KEY_PREFIX)
        print(f"  Redis cache flushed: {deleted} keys removed ({reason}).")
    except Exception:
        pass


def fetch_haproxy_stats(stats_url="http://localhost:8404/stats;csv"):
    """Fetch HAProxy stats CSV from the stats endpoint."""
    req = urllib.request.Request(stats_url)
    resp = urllib.request.urlopen(req, timeout=5)
    return resp.read().decode()


def parse_backend_sessions(stats_csv):
    """Sum current sessions (scur) for pg_backends backend rows."""
    total = 0
    reader = csv.DictReader(io.StringIO(stats_csv))
    for row in reader:
        pxname = (row.get("# pxname") or row.get("pxname") or "").strip()
        svname = (row.get("svname") or "").strip()
        if pxname == "pg_backends" and svname not in ("FRONTEND", "BACKEND"):
            try:
                total += int(row.get("scur", 0))
            except (ValueError, TypeError):
                pass
    return total


def parse_backend_queue(stats_csv):
    """Sum current queue depth (qcur) for pg_backends backend rows."""
    total = 0
    reader = csv.DictReader(io.StringIO(stats_csv))
    for row in reader:
        pxname = (row.get("# pxname") or row.get("pxname") or "").strip()
        svname = (row.get("svname") or "").strip()
        if pxname == "pg_backends" and svname == "BACKEND":
            try:
                total += int(row.get("qcur", 0))
            except (ValueError, TypeError):
                pass
    return total


def parse_frontend_session_total(stats_csv):
    """Return total session count (stot) for pg_frontend.

    Used to detect connection attempts even when all backends are DOWN
    (HAProxy doesn't queue in that case, but stot still increments).
    """
    reader = csv.DictReader(io.StringIO(stats_csv))
    for row in reader:
        pxname = (row.get("# pxname") or row.get("pxname") or "").strip()
        svname = (row.get("svname") or "").strip()
        if pxname == "pg_frontend" and svname == "FRONTEND":
            try:
                return int(row.get("stot", 0))
            except (ValueError, TypeError):
                return 0
    return 0


def cmd_scale_to(n):
    """Programmatic scale to N instances (used by idle-watch)."""
    class _Args:
        def __init__(self, count):
            self.count = count
            self.image = None
    cmd_scale(_Args(n))


def cmd_idle_watch(args):
    """Monitor HAProxy for idle connections and scale to zero / wake on connect."""
    idle_timeout = int(os.environ.get("IDLE_TIMEOUT_SECONDS", "900"))
    poll_interval = int(os.environ.get("IDLE_POLL_INTERVAL", "5"))
    stats_url = os.environ.get(
        "HAPROXY_STATS_URL", "http://localhost:8404/stats;csv"
    )
    last_active = time.time()
    current_replicas = get_instance_count()
    last_frontend_stot = None  # Track frontend session total for wake detection

    log = logging.getLogger("idle-watch")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [idle-watch] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    log.info(
        "Started: idle_timeout=%ds, poll=%ds, current_replicas=%d",
        idle_timeout, poll_interval, current_replicas,
    )

    while True:
        try:
            stats_csv = fetch_haproxy_stats(stats_url)
            active_sessions = parse_backend_sessions(stats_csv)
            queued = parse_backend_queue(stats_csv)
            frontend_stot = parse_frontend_session_total(stats_csv)
        except Exception as e:
            log.warning("Failed to fetch HAProxy stats: %s", e)
            time.sleep(poll_interval)
            continue

        # Detect new connection attempts via frontend stot counter.
        # When backends are DOWN, HAProxy doesn't queue (qcur=0) but
        # the frontend stot still increments on each connection attempt.
        new_connections = False
        if last_frontend_stot is not None and frontend_stot > last_frontend_stot:
            new_connections = True
        last_frontend_stot = frontend_stot

        if active_sessions > 0 or queued > 0 or new_connections:
            last_active = time.time()
            if current_replicas == 0:
                log.info(
                    "Wake-on-connect: sessions=%d queued=%d new_conns=%s, scaling up",
                    active_sessions, queued, new_connections,
                )
                try:
                    cmd_scale_to(1)
                    current_replicas = 1
                except Exception as e:
                    log.error("Scale-up failed: %s", e)
        elif time.time() - last_active > idle_timeout and current_replicas > 0:
            log.info("Idle for %ds, scaling to zero", idle_timeout)
            try:
                cmd_scale_to(0)
                current_replicas = 0
            except Exception as e:
                log.error("Scale-down failed: %s", e)

        time.sleep(poll_interval)


def cmd_cache_stats(args):
    """Show Redis cache statistics."""
    try:
        r = _redis_connection()
        info = r.info()
    except Exception as e:
        print(f"ERROR: Cannot connect to Redis: {e}")
        sys.exit(1)

    # Count only deployment-prefixed keys
    prefixed_count = 0
    cursor = 0
    while True:
        cursor, keys = r.scan(cursor=cursor, match=f"{REDIS_KEY_PREFIX}*", count=500)
        prefixed_count += len(keys)
        if cursor == 0:
            break

    print("=== Redis Cache Stats ===\n")
    print(f"  Memory used:     {info.get('used_memory_human', '?')}")
    print(f"  Peak memory:     {info.get('used_memory_peak_human', '?')}")
    print(f"  Cached queries:  {prefixed_count} (prefix: {REDIS_KEY_PREFIX})")
    print(f"  Hits:            {info.get('keyspace_hits', 0)}")
    print(f"  Misses:          {info.get('keyspace_misses', 0)}")
    hits = info.get('keyspace_hits', 0)
    misses = info.get('keyspace_misses', 0)
    total = hits + misses
    ratio = f"{hits / total * 100:.1f}%" if total > 0 else "N/A"
    print(f"  Hit ratio:       {ratio}")
    print(f"  Connected:       {info.get('connected_clients', '?')}")
    print(f"  Uptime:          {info.get('uptime_in_seconds', '?')}s")
    print(f"  Evicted keys:    {info.get('evicted_keys', 0)}")
    print(f"  Max memory:      {info.get('maxmemory_human', '?')}")
    print(f"  Eviction policy: {info.get('maxmemory_policy', '?')}")
    print()


def cmd_cache_flush(args):
    """Flush deployment-prefixed keys from Redis cache."""
    try:
        r = _redis_connection()
        deleted = _flush_redis_prefix(r, REDIS_KEY_PREFIX)
        print(f"Redis cache flushed: {deleted} keys removed (prefix: {REDIS_KEY_PREFIX}).")
    except Exception as e:
        print(f"ERROR: Cannot connect to Redis: {e}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="pg_deeplake serverless deployment management CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # status
    subparsers.add_parser("status", help="Show all instances, health, databases, connections")

    # health
    subparsers.add_parser("health", help="Detailed health check per instance")

    # provision-db
    p_db = subparsers.add_parser("provision-db", help="Create a database (serial DDL on instance 1)")
    p_db.add_argument("name", help="Database name to create")

    # run-ddl
    p_ddl = subparsers.add_parser("run-ddl", help="Execute DDL on instance 1")
    p_ddl.add_argument("sql", help="DDL statement to execute")
    p_ddl.add_argument("--database", "-d", default=None, help="Target database")

    # rotate-creds
    p_creds = subparsers.add_parser("rotate-creds", help="Rotate AWS credentials on all instances")
    p_creds.add_argument("--aws-access-key-id", default=None)
    p_creds.add_argument("--aws-secret-access-key", default=None)
    p_creds.add_argument("--aws-session-token", default=None)

    # scale
    p_scale = subparsers.add_parser("scale", help="Scale to N pg-deeplake instances")
    p_scale.add_argument("count", type=int, help="Number of instances (>= 0, 0 = scale-to-zero)")
    p_scale.add_argument("--image", default=None,
                         help=f"Docker image (default: $PG_DEEPLAKE_IMAGE or {DEFAULT_IMAGE})")

    # idle-watch
    subparsers.add_parser(
        "idle-watch",
        help="Monitor HAProxy and auto-scale to zero on idle / wake on connect",
    )

    # cache-stats
    subparsers.add_parser("cache-stats", help="Show Redis cache statistics")

    # cache-flush
    subparsers.add_parser("cache-flush", help="Flush Redis cache")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Commands that connect to PostgreSQL require a password.
    pg_commands = {"status", "health", "provision-db", "run-ddl", "rotate-creds"}
    if args.command in pg_commands:
        _require_password()

    commands = {
        "status": cmd_status,
        "health": cmd_health,
        "provision-db": cmd_provision_db,
        "run-ddl": cmd_run_ddl,
        "rotate-creds": cmd_rotate_creds,
        "scale": cmd_scale,
        "idle-watch": cmd_idle_watch,
        "cache-stats": cmd_cache_stats,
        "cache-flush": cmd_cache_flush,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
