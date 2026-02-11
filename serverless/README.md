# pg_deeplake Serverless Deployment

Serverless multi-instance pg_deeplake with HAProxy load balancing, shared S3 storage, and a management CLI.

## Architecture

```
Client -> HAProxy (TCP L4, :5432) -> pg-deeplake-1 (stateless, :5433)
                                  -> pg-deeplake-2 (stateless, :5434)
                                       |
                                    S3 (shared catalog + data)
                                       |
                                    Redis (:6379, L2 cache)
```

**Why HAProxy over PgBouncer**: pg_deeplake uses user-context GUCs (`SET deeplake.stateless_enabled`, `SET deeplake.root_path`, `SET deeplake.creds`). PgBouncer in transaction mode drops session state between transactions. HAProxy does TCP passthrough — transparent to the extension.

**Wake-on-connect**: HAProxy is configured with `timeout queue 120s` and `option redispatch`. When all backends are down (e.g. scale-to-zero in Kubernetes), incoming connections queue at HAProxy. When backends recover, queued connections are forwarded automatically. In the Helm chart, KEDA monitors HAProxy's queue depth via prometheus metrics to trigger scale-up from zero.

## Quick Start

```bash
# 1. Configure
cp .env.example .env
# Edit .env: add AWS credentials, set S3 path

# 2. Deploy
docker compose up -d

# 3. Verify
psql -h localhost -p 5432 -U postgres -c "SHOW deeplake.stateless_enabled"
# → on

python scripts/manage.py status
# → both instances UP, stateless on, same root_path

# 4. Create a database
python scripts/manage.py provision-db tpch

# 5. Run tests
pytest tests/test_deployment.py -v
```

## Components

| Service | Image | Port | Purpose |
|---------|-------|------|---------|
| pg-deeplake-1 | quay.io/activeloopai/pg-deeplake:18 | 5433 | Primary instance (DDL target) |
| pg-deeplake-2 | quay.io/activeloopai/pg-deeplake:18 | 5434 | Secondary instance |
| haproxy | haproxy:2.9 | 5432 (PG), 8404 (stats) | TCP L4 load balancer / activation proxy |
| redis | redis:7-alpine | 6379 | L2 query cache |

## Management CLI

```bash
python scripts/manage.py status        # Instance overview
python scripts/manage.py health        # Detailed health per instance
python scripts/manage.py provision-db NAME   # Create DB on all instances
python scripts/manage.py run-ddl "SQL" [-d DB]  # Execute DDL on instance 1 (guarded)
python scripts/manage.py run-ddl "SQL" --force  # Bypass DDL guard for advanced cases
python scripts/manage.py rotate-creds  # Update AWS creds on all instances
python scripts/manage.py scale N       # Scale to N instances
python scripts/manage.py cache-stats   # Redis cache statistics
python scripts/manage.py cache-flush   # Flush cache (deployment-prefixed keys only)
```

**Why DDL goes to instance 1 only**: pg_deeplake's S3 catalog doesn't support concurrent DDL. `run-ddl` serializes all DDL on instance 1; the sync worker propagates changes to other instances.

**Why provision-db hits all instances**: `CREATE DATABASE` is a PostgreSQL catalog operation that doesn't propagate via S3. `provision-db` creates the database on every instance.

## Scaling

```bash
# Scale to 4 instances
python scripts/manage.py scale 4

# Scale down to 1
python scripts/manage.py scale 1

# Scale back to default
python scripts/manage.py scale 2
```

The `scale` command generates `docker-compose.yml` and `config/haproxy.cfg` for the requested instance count, runs `docker compose up -d --remove-orphans`, waits for new instances to become healthy, and provisions existing databases on new instances.

## Configuration

### .env Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AWS_ACCESS_KEY_ID` | (required) | AWS access key for S3 |
| `AWS_SECRET_ACCESS_KEY` | (required) | AWS secret key |
| `AWS_SESSION_TOKEN` | (optional) | STS session token |
| `DEEPLAKE_ROOT_PATH` | (required) | S3 path for data + catalog |
| `DEEPLAKE_SYNC_INTERVAL_MS` | 2000 | Catalog sync polling interval |
| `PG_DEEPLAKE_MEMORY_LIMIT_MB` | 0 | Memory limit (0 = unlimited) |
| `DEEPLAKE_STARTUP_JITTER_MAX_SECONDS` | 0 | Random startup delay (0..N sec) to reduce thundering-herd scale-ups |
| `POSTGRES_USER` | postgres | PostgreSQL superuser |
| `POSTGRES_PASSWORD` | postgres | PostgreSQL password |
| `COMPOSE_PROFILES` | haproxy | Load balancer profile (`haproxy`) |

### PostgreSQL Tuning (postgresql-overrides.conf)

Optimized for ephemeral instances where local disk is disposable:
- `wal_level=minimal`, `fsync=off` — no durability needed (data is in S3)
- `shared_buffers=512MB`, `work_mem=2GB` — analytical workloads
- `max_connections=200`
- DDL + slow query (>1s) logging

> **Note**: When deploying to Kubernetes with persistent volumes (`persistence.enabled=true` in Helm), the chart automatically uses safe WAL settings (`fsync=on`, `synchronous_commit=on`).

### TLS

For local development, TLS is off by default. To enable:

1. Place a combined cert+key PEM at `config/certs/server.pem`
2. HAProxy will automatically use SSL for the frontend bind

For Kubernetes, TLS is on by default in the Helm chart. Create the default TLS secret:
```bash
kubectl create secret tls pg-deeplake-tls --cert=server.crt --key=server.key
helm install pg-deeplake ./helm/pg-deeplake
```

### HAProxy (haproxy.cfg)

- TCP mode with `pgsql-check` health checks
- Round-robin across backends
- 1-hour client/server timeouts (long COPY/analytical queries)
- `timeout queue 120s` — connections queue when all backends are down
- `option redispatch` — retry on available server after failure
- Stats UI: disabled by default (enable via `HAPROXY_STATS_USER` + `HAPROXY_STATS_PASSWORD`)
- Prometheus: http://localhost:8404/metrics

### Monitoring

```bash
# HAProxy stats UI (if enabled)
curl -u "$HAPROXY_STATS_USER:$HAPROXY_STATS_PASSWORD" http://localhost:8404/stats

# Prometheus metrics
curl http://localhost:8404/metrics

# Container health
docker ps

# Instance health
python scripts/manage.py health

# Redis cache stats
python scripts/manage.py cache-stats
```

## Kubernetes Deployment (Helm)

```bash
helm install pg-deeplake ./helm/pg-deeplake \
  --set aws.accessKeyId=AKIA... \
  --set aws.secretAccessKey=... \
  --set deeplake.rootPath=s3://bucket/prefix/ \
  --set tls.secretName=pg-deeplake-tls

kubectl get pods  # N pg-deeplake pods + haproxy + redis
```

### Scale-to-Zero

The Helm chart supports KEDA-driven scale-to-zero:

```bash
helm install pg-deeplake ./helm/pg-deeplake \
  --set autoscaling.enabled=true \
  --set autoscaling.minReplicas=0 \
  --set autoscaling.maxReplicas=8
```

**How it works**: HAProxy is the always-on activation proxy. At 0 DB replicas, new client connections queue at HAProxy. KEDA monitors HAProxy's `haproxy_backend_current_queue` prometheus metric. When the queue rises above the threshold, KEDA scales the StatefulSet from 0 to 1. Once the DB pod passes health checks, HAProxy forwards the queued connections.

## Data Ingestion (TPC-H)

```bash
# Through HAProxy (round-robin to any instance)
python ../tpch_deeplake_ingest.py --port 5432 --stateless \
    --s3-root-path s3://your-bucket/path/ \
    --data-dir /path/to/tpch_data

# Direct to instance 1 (DDL safety)
python ../tpch_deeplake_ingest.py --port 5433 --stateless \
    --s3-root-path s3://your-bucket/path/
```

## Known Upstream Bugs

The following are pg_deeplake extension bugs being tracked upstream. They are **not** expected behavior and should be fixed in future extension releases.

| Bug | Impact | Current Mitigation |
|-----|--------|-------------------|
| Sync worker SIGPIPE (exit 141) | Instance restarts briefly on first table sync | `restart: unless-stopped` + HAProxy failover |
| Concurrent DDL deadlocks | Catalog corruption | Serialized DDL via `manage.py run-ddl` |
| DROP TABLE/DATABASE crash | Extension process crash | Avoid DROP; use fresh S3 paths |
| VARCHAR(1) catalog mismatch | Wrong type on synced instance | Use latest pg_deeplake image |
| OOM on large COPY | Process killed | Set `PG_DEEPLAKE_MEMORY_LIMIT_MB`; chunked COPY |

Tests include retry logic and extended sync waits to work around these bugs. The test harness is designed to be resilient to transient crashes, but the underlying issues require upstream fixes.

## STS Credential Rotation

AWS STS tokens expire. Rotate without restarting:

```bash
# Update .env with new credentials, then:
python scripts/manage.py rotate-creds

# Or pass directly:
python scripts/manage.py rotate-creds \
    --aws-access-key-id NEW_KEY \
    --aws-secret-access-key NEW_SECRET \
    --aws-session-token NEW_TOKEN
```

If rotation fails on any instance, the command exits with a non-zero status and warns about potentially inconsistent credentials.

## Runbooks

Operational runbooks are in `serverless/RUNBOOKS.md`:
- credential rotation
- backup/restore workflow
- incident playbooks (OOM, queue buildup, pod crash)

## Monitoring Assets

Monitoring references are in `serverless/monitoring/`:
- `serverless/monitoring/grafana-dashboard.json`
- `serverless/monitoring/README.md`
