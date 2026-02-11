# Serverless Runbooks

## Credential Rotation

1. Obtain fresh credentials (or update secret source for IRSA/external secrets).
2. Validate connectivity to all instances:
   `python scripts/manage.py health`
3. Rotate credentials:
   `python scripts/manage.py rotate-creds`
4. Confirm reload succeeded on all instances and verify read/write queries.
5. Flush deployment cache if needed:
   `python scripts/manage.py cache-flush`

## Backup and Restore

1. Backup object storage prefix that holds Deep Lake datasets and catalog.
2. Backup Kubernetes manifests/Helm values used for deployment.
3. Backup PostgreSQL control-plane catalogs if local metadata is used.
4. Restore by deploying chart with the same `deeplake.rootPath` and credentials.
5. Validate recovered tables with `python scripts/manage.py health`.

## Incident Playbooks

### OOM Restarts

1. Check `PgDeeplakeOOMKilled` alert details.
2. Increase `resources.limits.memory` and/or `PG_DEEPLAKE_MEMORY_LIMIT_MB`.
3. Restart affected pods and watch query success rate.

### HAProxy Queue Buildup

1. Check `PgDeeplakeHighQueueDepth` and backend pod readiness.
2. Scale out DB replicas and verify HAProxy backend health.
3. Investigate slow queries and cache hit ratio.

### Pod Crash or NotReady

1. Inspect pod logs and `health-check.sh` output.
2. Validate credentials/root path configuration.
3. If needed, roll pod and verify recovery via `manage.py health`.
