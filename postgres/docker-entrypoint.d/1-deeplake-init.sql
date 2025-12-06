-- Query planner settings
ALTER SYSTEM SET enable_hashjoin TO on;
ALTER SYSTEM SET enable_mergejoin TO off;
ALTER SYSTEM SET enable_nestloop TO off;

-- Memory settings
ALTER SYSTEM SET work_mem TO '2GB';

-- Init PgDeeplake
CREATE EXTENSION pg_deeplake;
