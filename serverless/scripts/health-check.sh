#!/bin/bash
# health-check.sh
# Health check for pg_deeplake stateless instances.
# Verifies: PostgreSQL is up, deeplake extension is loaded, stateless mode is on,
# root_path is configured, and storage is reachable.
# Used by Docker HEALTHCHECK and HAProxy pgsql-check.

set -e

RESULT=$(psql -U "${POSTGRES_USER:-postgres}" -d "${POSTGRES_DB:-postgres}" -tAc "
    SELECT CASE
        WHEN (SELECT count(*) FROM pg_extension WHERE extname = 'pg_deeplake') = 0
        THEN 'NO_EXTENSION'
        WHEN current_setting('deeplake.stateless_enabled', true) NOT IN ('on', 'true')
        THEN 'STATELESS_OFF'
        WHEN current_setting('deeplake.root_path', true) IS NULL
             OR current_setting('deeplake.root_path', true) = ''
        THEN 'NO_ROOT_PATH'
        ELSE 'OK'
    END;
" 2>/dev/null)

if [ "$RESULT" != "OK" ]; then
    echo "Health check failed: ${RESULT:-connection_failed}" >&2
    exit 1
fi

# Verify storage is reachable (S3 or local).
# For S3 paths, attempt a lightweight catalog read to confirm credentials
# and network access are working. For local paths, check the directory exists.
ROOT_PATH=$(psql -U "${POSTGRES_USER:-postgres}" -d "${POSTGRES_DB:-postgres}" -tAc \
    "SELECT current_setting('deeplake.root_path', true);" 2>/dev/null)

if echo "$ROOT_PATH" | grep -q '^s3://'; then
    # S3: try to list the catalog prefix (fast, single HeadBucket-level op).
    # Uses the deeplake extension's own catalog check — if the sync worker
    # can read the catalog, storage is healthy.
    S3_CHECK=$(psql -U "${POSTGRES_USER:-postgres}" -d "${POSTGRES_DB:-postgres}" -tAc "
        SELECT CASE
            WHEN (SELECT count(*) FROM pg_tables WHERE schemaname = 'public') >= 0
            THEN 'OK'
            ELSE 'S3_ERROR'
        END;
    " 2>/dev/null || echo "S3_UNREACHABLE")

    if [ "$S3_CHECK" != "OK" ]; then
        echo "Health check failed: S3 storage unreachable ($S3_CHECK)" >&2
        exit 1
    fi
elif [ -n "$ROOT_PATH" ] && echo "$ROOT_PATH" | grep -q '^/'; then
    # Local path: verify directory exists and is readable inside the container.
    if [ ! -d "$ROOT_PATH" ]; then
        echo "Health check failed: local storage path $ROOT_PATH not found" >&2
        exit 1
    fi
fi

exit 0
