#!/usr/bin/env python3
"""
query_cache.py — Redis read-through cache for pg_deeplake queries.

Wraps psycopg2 connections with a Redis L2 cache. SELECT queries are
hashed and cached; writes (INSERT/UPDATE/DELETE/DDL) bypass the cache.

Usage:
    from query_cache import CachedConnection

    conn = CachedConnection(
        host="localhost", port=5432, database="postgres",
        user="postgres", password="postgres",
        redis_host="localhost", redis_port=6379,
        ttl=60,
    )
    rows = conn.execute("SELECT * FROM my_table WHERE id = %s", (42,))
    conn.close()
"""

import hashlib
import json
import re

import psycopg2

# Patterns that indicate a write/DDL statement (skip caching)
_WRITE_PATTERN = re.compile(
    r"^\s*(INSERT|UPDATE|DELETE|CREATE|DROP|ALTER|TRUNCATE|GRANT|REVOKE|"
    r"VACUUM|ANALYZE|REINDEX|CLUSTER|COPY|SET|RESET|DISCARD|BEGIN|"
    r"COMMIT|ROLLBACK|SAVEPOINT|RELEASE|PREPARE|EXECUTE|DEALLOCATE|"
    r"LISTEN|NOTIFY|UNLISTEN|LOCK)\b",
    re.IGNORECASE,
)


class CachedConnection:
    """PostgreSQL connection with Redis read-through caching."""

    def __init__(
        self,
        host="localhost",
        port=5432,
        database="postgres",
        user="postgres",
        password="postgres",
        redis_host="localhost",
        redis_port=6379,
        ttl=60,
        key_prefix="pgdl:",
    ):
        self._pg = psycopg2.connect(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password,
        )
        self._pg.autocommit = True
        self._ttl = ttl
        self._prefix = key_prefix

        try:
            import redis
            self._redis = redis.Redis(
                host=redis_host, port=redis_port, decode_responses=True
            )
            self._redis.ping()
        except Exception:
            self._redis = None

    def _cache_key(self, sql_str, params):
        """Build a Redis key from SQL + params."""
        raw = sql_str + "|" + json.dumps(params, default=str, sort_keys=True)
        digest = hashlib.sha256(raw.encode()).hexdigest()[:16]
        return f"{self._prefix}{digest}"

    def _is_read(self, sql_str):
        """Return True if the query is a read (SELECT/SHOW/EXPLAIN/WITH)."""
        return not _WRITE_PATTERN.match(sql_str.strip())

    def execute(self, sql_str, params=None):
        """Execute a query, returning rows for reads (cached) or None for writes."""
        params = params or ()

        if self._is_read(sql_str) and self._redis:
            key = self._cache_key(sql_str, params)
            try:
                cached = self._redis.get(key)
                if cached is not None:
                    return json.loads(cached)
            except Exception:
                pass

        with self._pg.cursor() as cur:
            cur.execute(sql_str, params)
            if cur.description is None:
                return None
            rows = cur.fetchall()

        if self._is_read(sql_str) and self._redis:
            try:
                self._redis.setex(key, self._ttl, json.dumps(rows, default=str))
            except Exception:
                pass

        return rows

    def close(self):
        """Close the PostgreSQL connection."""
        self._pg.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
