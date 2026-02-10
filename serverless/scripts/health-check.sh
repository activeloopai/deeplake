#!/bin/bash
# health-check.sh
# Health check for pg_deeplake stateless instances.
# Verifies: PostgreSQL is up, deeplake extension is loaded, stateless mode is on,
# root_path is configured to S3.
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

if [ "$RESULT" = "OK" ]; then
    exit 0
else
    echo "Health check failed: ${RESULT:-connection_failed}" >&2
    exit 1
fi
