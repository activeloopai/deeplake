#!/usr/bin/env python3
"""
TPC-H Ingestion Script for pg_deeplake

Each table gets its own connection through a load balancer (HAProxy),
which distributes tables across backend instances via round-robin.

Usage:
    # Via HAProxy (parallel, one connection per table):
    python tpch_deeplake_ingest.py

    # Direct to single instance:
    python tpch_deeplake_ingest.py --port 5433 --sequential
"""

import argparse
import io
import sys
import time
import psycopg2
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def log(msg):
    print(msg, flush=True)

# TPC-H table definitions
TPCH_TABLES = {
    'region': {
        'columns': [
            ('r_regionkey', 'INTEGER'),
            ('r_name', 'VARCHAR(25)'),
            ('r_comment', 'VARCHAR(152)')
        ],
    },
    'nation': {
        'columns': [
            ('n_nationkey', 'INTEGER'),
            ('n_name', 'VARCHAR(25)'),
            ('n_regionkey', 'INTEGER'),
            ('n_comment', 'VARCHAR(152)')
        ],
    },
    'supplier': {
        'columns': [
            ('s_suppkey', 'INTEGER'),
            ('s_name', 'VARCHAR(25)'),
            ('s_address', 'VARCHAR(40)'),
            ('s_nationkey', 'INTEGER'),
            ('s_phone', 'VARCHAR(15)'),
            ('s_acctbal', 'DECIMAL(15,2)'),
            ('s_comment', 'VARCHAR(101)')
        ],
    },
    'customer': {
        'columns': [
            ('c_custkey', 'INTEGER'),
            ('c_name', 'VARCHAR(25)'),
            ('c_address', 'VARCHAR(40)'),
            ('c_nationkey', 'INTEGER'),
            ('c_phone', 'VARCHAR(15)'),
            ('c_acctbal', 'DECIMAL(15,2)'),
            ('c_mktsegment', 'VARCHAR(10)'),
            ('c_comment', 'VARCHAR(117)')
        ],
    },
    'part': {
        'columns': [
            ('p_partkey', 'INTEGER'),
            ('p_name', 'VARCHAR(55)'),
            ('p_mfgr', 'VARCHAR(25)'),
            ('p_brand', 'VARCHAR(10)'),
            ('p_type', 'VARCHAR(25)'),
            ('p_size', 'INTEGER'),
            ('p_container', 'VARCHAR(10)'),
            ('p_retailprice', 'DECIMAL(15,2)'),
            ('p_comment', 'VARCHAR(23)')
        ],
    },
    'partsupp': {
        'columns': [
            ('ps_partkey', 'INTEGER'),
            ('ps_suppkey', 'INTEGER'),
            ('ps_availqty', 'INTEGER'),
            ('ps_supplycost', 'DECIMAL(15,2)'),
            ('ps_comment', 'VARCHAR(199)')
        ],
    },
    'orders': {
        'columns': [
            ('o_orderkey', 'INTEGER'),
            ('o_custkey', 'INTEGER'),
            ('o_orderstatus', 'VARCHAR(1)'),
            ('o_totalprice', 'DECIMAL(15,2)'),
            ('o_orderdate', 'DATE'),
            ('o_orderpriority', 'VARCHAR(15)'),
            ('o_clerk', 'VARCHAR(15)'),
            ('o_shippriority', 'INTEGER'),
            ('o_comment', 'VARCHAR(79)')
        ],
    },
    'lineitem': {
        'columns': [
            ('l_orderkey', 'INTEGER'),
            ('l_partkey', 'INTEGER'),
            ('l_suppkey', 'INTEGER'),
            ('l_linenumber', 'INTEGER'),
            ('l_quantity', 'DECIMAL(15,2)'),
            ('l_extendedprice', 'DECIMAL(15,2)'),
            ('l_discount', 'DECIMAL(15,2)'),
            ('l_tax', 'DECIMAL(15,2)'),
            ('l_returnflag', 'VARCHAR(1)'),
            ('l_linestatus', 'VARCHAR(1)'),
            ('l_shipdate', 'DATE'),
            ('l_commitdate', 'DATE'),
            ('l_receiptdate', 'DATE'),
            ('l_shipinstruct', 'VARCHAR(25)'),
            ('l_shipmode', 'VARCHAR(10)'),
            ('l_comment', 'VARCHAR(44)')
        ],
    },
}

TABLE_LOAD_ORDER = ['region', 'nation', 'supplier', 'customer', 'part', 'partsupp', 'orders', 'lineitem']


def get_connection(host, port, database, user, password):
    return psycopg2.connect(host=host, port=port, database=database, user=user, password=password)



def disable_autovacuum(conn):
    old_autocommit = conn.autocommit
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute("ALTER SYSTEM SET autovacuum = off;")
            cur.execute("SELECT pg_reload_conf();")
    finally:
        conn.autocommit = old_autocommit


def enable_autovacuum(conn):
    old_autocommit = conn.autocommit
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute("ALTER SYSTEM SET autovacuum = on;")
            cur.execute("SELECT pg_reload_conf();")
    finally:
        conn.autocommit = old_autocommit


def drop_table(conn, table_name):
    with conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE;")
    conn.commit()


def create_table(conn, table_name, table_def):
    columns = table_def['columns']
    col_defs = ', '.join([f"{name} {dtype}" for name, dtype, *_ in columns])
    sql = f"CREATE TABLE {table_name} ({col_defs}) USING deeplake;"
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()


def table_exists(conn, table_name):
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT 1
            FROM pg_tables
            WHERE schemaname = 'public' AND tablename = %s
            """,
            (table_name,),
        )
        return cur.fetchone() is not None


CHUNK_SIZE = 5_000_000  # rows per COPY batch to avoid OOM on large tables


def load_data(conn, table_name, data_file):
    """Load data in chunks to avoid OOM on large tables like lineitem (60M rows).

    Each chunk is a separate COPY + COMMIT so deeplake flushes to S3
    before accumulating the next chunk.
    """
    total_lines = 0
    chunk_num = 0

    with open(data_file, 'r') as f:
        while True:
            chunk = io.StringIO()
            lines_in_chunk = 0

            for line in f:
                if line.endswith('|\n'):
                    chunk.write(line[:-2] + '\n')
                elif line.endswith('|'):
                    chunk.write(line[:-1])
                else:
                    chunk.write(line)
                lines_in_chunk += 1
                if lines_in_chunk >= CHUNK_SIZE:
                    break

            if lines_in_chunk == 0:
                break

            chunk.seek(0)
            with conn.cursor() as cur:
                cur.copy_expert(
                    f"COPY {table_name} FROM STDIN WITH (FORMAT csv, DELIMITER '|')",
                    chunk
                )
            conn.commit()

            total_lines += lines_in_chunk
            chunk_num += 1
            if chunk_num > 1 or lines_in_chunk >= CHUNK_SIZE:
                log(f"    chunk {chunk_num}: {lines_in_chunk:,} rows committed ({total_lines:,} total)")

    return total_lines


def get_row_count(conn, table_name):
    with conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) FROM {table_name};")
        return cur.fetchone()[0]


def run_vacuum(conn, table_name):
    old_autocommit = conn.autocommit
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute(f"VACUUM ANALYZE {table_name};")
    finally:
        conn.autocommit = old_autocommit


def ingest_one_table(table_name, args):
    """Worker: open a new connection (via LB), load one table. Returns (table, rows, seconds)."""
    conn = get_connection(args.host, args.port, args.database, args.user, args.password)
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("SELECT inet_server_addr()")
            backend_addr = cur.fetchone()[0]
        conn.autocommit = False

        data_file = Path(args.data_dir) / f"{table_name}.tbl"
        if not data_file.exists():
            log(f"[{table_name}] Data file not found: {data_file}, skipping")
            return (table_name, 0, 0.0)

        log(f"[{table_name}] Loading on backend {backend_addr}...")

        start_time = time.time()
        load_data(conn, table_name, data_file)
        elapsed = time.time() - start_time

        row_count = get_row_count(conn, table_name)
        log(f"[{table_name}] {row_count:,} rows in {elapsed:.1f}s (backend {backend_addr})")

        if args.vacuum_after_each:
            run_vacuum(conn, table_name)

        return (table_name, row_count, elapsed)
    except Exception as e:
        log(f"[{table_name}] ERROR: {e}")
        raise
    finally:
        try:
            conn.rollback()
        except Exception:
            pass
        conn.close()


def main():
    parser = argparse.ArgumentParser(description='TPC-H pg_deeplake ingestion')

    # Connection
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--port', type=int, default=5432,
                        help='Port (default: 5432 for HAProxy)')
    parser.add_argument('--database', default='postgres')
    parser.add_argument('--user', default='postgres')
    parser.add_argument('--password', default='postgres')

    # Ingestion
    parser.add_argument('--data-dir', default='/home/admin/sasun/work/tpch/tpch_data')
    parser.add_argument('--tables', nargs='*', default=None,
                        help='Specific tables to load (default: all 8)')
    parser.add_argument('--skip-create', action='store_true')
    parser.add_argument('--replace-existing', action='store_true',
                        help='DROP+CREATE existing tables before load')
    parser.add_argument('--sequential', action='store_true',
                        help='Load tables sequentially on one connection (no parallel)')
    parser.add_argument('--disable-autovacuum', action='store_true')
    parser.add_argument('--vacuum-after-each', action='store_true')

    args = parser.parse_args()

    tables_to_load = args.tables if args.tables else TABLE_LOAD_ORDER

    for t in tables_to_load:
        if t not in TPCH_TABLES:
            parser.error(f"Unknown table: {t}")

    log(f"TPC-H pg_deeplake Ingestion")
    log(f"  Endpoint:   {args.host}:{args.port}")
    log(f"  Database:   {args.database}")
    log(f"  Tables:     {', '.join(tables_to_load)}")
    log(f"  Parallel:   {not args.sequential} ({len(tables_to_load)} connections)")
    log("")

    # Phase 1: Create tables serially on one connection
    if not args.skip_create:
        log("Phase 1: Creating tables...")
        conn = get_connection(args.host, args.port, args.database, args.user, args.password)
        try:
            if args.disable_autovacuum:
                disable_autovacuum(conn)
            for table_name in tables_to_load:
                table_def = TPCH_TABLES[table_name]
                if args.replace_existing:
                    drop_table(conn, table_name)
                    create_table(conn, table_name, table_def)
                    log(f"  {table_name}: recreated")
                elif not table_exists(conn, table_name):
                    create_table(conn, table_name, table_def)
                    log(f"  {table_name}: created")
                else:
                    log(f"  {table_name}: exists")
        finally:
            try:
                conn.rollback()
            except Exception:
                pass
            conn.close()

        log("")

    # Phase 2: Load data
    overall_start = time.time()

    if args.sequential:
        # Sequential: one connection, all tables
        log("Loading data sequentially...")
        results = []
        for table_name in tables_to_load:
            result = ingest_one_table(table_name, args)
            results.append(result)
    else:
        # Parallel: one connection per table, HAProxy distributes
        log(f"Phase 2: Loading data in parallel ({len(tables_to_load)} tables, 1 connection each)...")
        results = []
        with ThreadPoolExecutor(max_workers=len(tables_to_load)) as executor:
            futures = {
                executor.submit(ingest_one_table, table_name, args): table_name
                for table_name in tables_to_load
            }
            for future in as_completed(futures):
                table_name = futures[future]
                try:
                    results.append(future.result())
                except Exception as e:
                    log(f"[{table_name}] FAILED: {e}")

    overall_elapsed = time.time() - overall_start
    total_rows = sum(r[1] for r in results)

    log(f"\nDone. {total_rows:,} total rows in {overall_elapsed:.1f}s (wall time)")
    for table_name, row_count, elapsed in sorted(results, key=lambda x: x[0]):
        log(f"  {table_name:<12} {row_count:>12,} rows  {elapsed:>7.1f}s")


if __name__ == '__main__':
    main()
