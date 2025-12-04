\i sql/utils.psql

DROP EXTENSION IF EXISTS pg_deeplake CASCADE;
CREATE EXTENSION pg_deeplake;

\i sql/tpch/create_schema.sql

DO $$
BEGIN
    RAISE NOTICE 'Schema created';
END
$$;

\i sql/tpch/insert.sql

DO $$
BEGIN
RAISE NOTICE 'Data inserted, creating indexes...';
CREATE UNIQUE INDEX idx_region_pk    ON region(r_regionkey);
CREATE UNIQUE INDEX idx_nation_pk    ON nation(n_nationkey);
CREATE UNIQUE INDEX idx_supplier_pk  ON supplier(s_suppkey);
CREATE UNIQUE INDEX idx_customer_pk  ON customer(c_custkey);
CREATE UNIQUE INDEX idx_orders_pk    ON orders(o_orderkey);
CREATE UNIQUE INDEX idx_part_pk      ON part(p_partkey);
CREATE UNIQUE INDEX idx_partsupp_pk  ON partsupp(ps_partkey, ps_suppkey);
CREATE UNIQUE INDEX idx_lineitem_pk  ON lineitem(l_orderkey, l_linenumber);
CREATE INDEX idx_deeplake_part_brand_container ON part USING btree (p_brand, p_container);
CREATE INDEX idx_deeplake_lineitem_partkey_qty ON lineitem USING btree (l_partkey, l_quantity);
RAISE NOTICE 'Running TPC-H queries...';
END
$$;

-- Uncomment to run PG default executor
-- SET pg_deeplake.use_deeplake_executor = off;

\echo '========================================='
\echo 'TPC-H Query 1'
\echo '========================================='
\i sql/tpch/1.sql
SELECT CASE WHEN :ROW_COUNT = 4 THEN 'true' ELSE 'false' END AS q1_passed \gset
\if :q1_passed
    \echo 'Query 1: PASSED'
\else
    DROP EXTENSION pg_deeplake CASCADE;
    DO $$
    BEGIN
        RAISE EXCEPTION 'Query 1 FAILED: Expected 4 rows';
    END
    $$;
    \quit
\endif


\echo '========================================='
\echo 'TPC-H Query 2'
\echo '========================================='
\i sql/tpch/2.sql
SELECT CASE WHEN :ROW_COUNT = 10 THEN 'true' ELSE 'false' END AS q2_passed \gset
\if :q2_passed
    \echo 'Query 2: PASSED'
\else
    DROP EXTENSION pg_deeplake CASCADE;
    DO $$
    BEGIN
        RAISE EXCEPTION 'Query 2 FAILED: Expected 10 rows';
    END
    $$;
    \quit
\endif

\echo '========================================='
\echo 'TPC-H Query 3'
\echo '========================================='
\i sql/tpch/3.sql
SELECT CASE WHEN :ROW_COUNT = 10 THEN 'true' ELSE 'false' END AS q3_passed \gset
\if :q3_passed
    \echo 'Query 3: PASSED'
\else
    DROP EXTENSION pg_deeplake CASCADE;
    DO $$
    BEGIN
        RAISE EXCEPTION 'Query 3 FAILED: Expected 10 row';
    END
    $$;
    \quit
\endif

\echo '========================================='
\echo 'TPC-H Query 4'
\echo '========================================='
\i sql/tpch/4.sql
SELECT CASE WHEN :ROW_COUNT = 5 THEN 'true' ELSE 'false' END AS q4_passed \gset
\if :q4_passed
    \echo 'Query 4: PASSED'
\else
    DROP EXTENSION pg_deeplake CASCADE;
    DO $$
    BEGIN
        RAISE EXCEPTION 'Query 4 FAILED: Expected 5 rows';
    END
    $$;
    \quit
\endif

\echo '========================================='
\echo 'TPC-H Query 5'
\echo '========================================='
\i sql/tpch/5.sql
SELECT CASE WHEN :ROW_COUNT = 5 THEN 'true' ELSE 'false' END AS q5_passed \gset
\if :q5_passed
    \echo 'Query 5: PASSED'
\else
    DROP EXTENSION pg_deeplake CASCADE;
    DO $$
    BEGIN
        RAISE EXCEPTION 'Query 5 FAILED: Expected 5 rows';
    END
    $$;
    \quit
\endif

\echo '========================================='
\echo 'TPC-H Query 6'
\echo '========================================='
\i sql/tpch/6.sql
SELECT CASE WHEN :ROW_COUNT = 1 THEN 'true' ELSE 'false' END AS q6_passed \gset
\if :q6_passed
    \echo 'Query 6: PASSED'
\else
    DROP EXTENSION pg_deeplake CASCADE;
    DO $$
    BEGIN
        RAISE EXCEPTION 'Query 6 FAILED: Expected 1 rows';
    END
    $$;
    \quit
\endif

\echo '========================================='
\echo 'TPC-H Query 7'
\echo '========================================='
\i sql/tpch/7.sql
SELECT CASE WHEN :ROW_COUNT = 4 THEN 'true' ELSE 'false' END AS q7_passed \gset
\if :q7_passed
    \echo 'Query 7: PASSED'
\else
    DROP EXTENSION pg_deeplake CASCADE;
    DO $$
    BEGIN
        RAISE EXCEPTION 'Query 7 FAILED: Expected 4 rows';
    END
    $$;
    \quit
\endif

\echo '========================================='
\echo 'TPC-H Query 8'
\echo '========================================='
\i sql/tpch/8.sql
SELECT CASE WHEN :ROW_COUNT = 1 THEN 'true' ELSE 'false' END AS q8_passed \gset
\if :q8_passed
    \echo 'Query 8: PASSED'
\else
    DROP EXTENSION pg_deeplake CASCADE;
    DO $$
    BEGIN
        RAISE EXCEPTION 'Query 8 FAILED: Expected 1 row';
    END
    $$;
    \quit
\endif

\echo '========================================='
\echo 'TPC-H Query 9'
\echo '========================================='
\i sql/tpch/9.sql
SELECT CASE WHEN :ROW_COUNT = 151 THEN 'true' ELSE 'false' END AS q9_passed \gset
\if :q9_passed
    \echo 'Query 9: PASSED'
\else
    DROP EXTENSION pg_deeplake CASCADE;
    DO $$
    BEGIN
        RAISE EXCEPTION 'Query 9 FAILED: Expected 151 rows';
    END
    $$;
    \quit
\endif

\echo '========================================='
\echo 'TPC-H Query 10'
\echo '========================================='
\i sql/tpch/10.sql
SELECT CASE WHEN :ROW_COUNT = 20 THEN 'true' ELSE 'false' END AS q10_passed \gset
\if :q10_passed
    \echo 'Query 10: PASSED'
\else
    DROP EXTENSION pg_deeplake CASCADE;
    DO $$
    BEGIN
        RAISE EXCEPTION 'Query 10 FAILED: Expected 20 rows';
    END
    $$;
    \quit
\endif

\echo '========================================='
\echo 'TPC-H Query 11'
\echo '========================================='
\i sql/tpch/11.sql
SELECT CASE WHEN :ROW_COUNT = 3127 THEN 'true' ELSE 'false' END AS q11_passed \gset
\if :q11_passed
    \echo 'Query 11: PASSED'
\else
    DROP EXTENSION pg_deeplake CASCADE;
    DO $$
    BEGIN
        RAISE EXCEPTION 'Query 11 FAILED: Expected 3127 rows';
    END
    $$;
    \quit
\endif

\echo '========================================='
\echo 'TPC-H Query 12'
\echo '========================================='
\i sql/tpch/12.sql
SELECT CASE WHEN :ROW_COUNT = 2 THEN 'true' ELSE 'false' END AS q12_passed \gset
\if :q12_passed
    \echo 'Query 12: PASSED'
\else
    DROP EXTENSION pg_deeplake CASCADE;
    DO $$
    BEGIN
        RAISE EXCEPTION 'Query 12 FAILED: Expected 2 rows';
    END
    $$;
    \quit
\endif

\echo '========================================='
\echo 'TPC-H Query 13'
\echo '========================================='
\i sql/tpch/13.sql
SELECT CASE WHEN :ROW_COUNT = 6 THEN 'true' ELSE 'false' END AS q13_passed \gset
\if :q13_passed
    \echo 'Query 13: PASSED'
\else
    DROP EXTENSION pg_deeplake CASCADE;
    DO $$
    BEGIN
        RAISE EXCEPTION 'Query 13 FAILED: Expected 6 rows';
    END
    $$;
    \quit
\endif

\echo '========================================='
\echo 'TPC-H Query 14'
\echo '========================================='
\i sql/tpch/14.sql
SELECT CASE WHEN :ROW_COUNT = 1 THEN 'true' ELSE 'false' END AS q14_passed \gset
\if :q14_passed
    \echo 'Query 14: PASSED'
\else
    DROP EXTENSION pg_deeplake CASCADE;
    DO $$
    BEGIN
        RAISE EXCEPTION 'Query 14 FAILED: Expected 1 row';
    END
    $$;
    \quit
\endif

\echo '========================================='
\echo 'TPC-H Query 15'
\echo '========================================='
\i sql/tpch/15.sql
SELECT CASE WHEN :ROW_COUNT = 1 THEN 'true' ELSE 'false' END AS q15_passed \gset
\if :q15_passed
    \echo 'Query 15: PASSED'
\else
    DROP EXTENSION pg_deeplake CASCADE;
    DO $$
    BEGIN
        RAISE EXCEPTION 'Query 15 FAILED: Expected 1 row';
    END
    $$;
    \quit
\endif

\echo '========================================='
\echo 'TPC-H Query 16'
\echo '========================================='
\i sql/tpch/16.sql
SELECT CASE WHEN :ROW_COUNT = 10 THEN 'true' ELSE 'false' END AS q16_passed \gset
\if :q16_passed
    \echo 'Query 16: PASSED'
\else
    DROP EXTENSION pg_deeplake CASCADE;
    DO $$
    BEGIN
        RAISE EXCEPTION 'Query 16 FAILED: Expected 10 rows';
    END
    $$;
    \quit
\endif

\echo '========================================='
\echo 'TPC-H Query 17'
\echo '========================================='
\i sql/tpch/17.sql
SELECT CASE WHEN :ROW_COUNT = 1 THEN 'true' ELSE 'false' END AS q17_passed \gset
\if :q17_passed
    \echo 'Query 17: PASSED'
\else
    DROP EXTENSION pg_deeplake CASCADE;
    DO $$
    BEGIN
        RAISE EXCEPTION 'Query 17 FAILED: Expected 1 row';
    END
    $$;
    \quit
\endif

\echo '========================================='
\echo 'TPC-H Query 18'
\echo '========================================='
\i sql/tpch/18.sql
SELECT CASE WHEN :ROW_COUNT = 7 THEN 'true' ELSE 'false' END AS q18_passed \gset
\if :q18_passed
    \echo 'Query 18: PASSED'
\else
    DROP EXTENSION pg_deeplake CASCADE;
    DO $$
    BEGIN
        RAISE EXCEPTION 'Query 18 FAILED: Expected 7 rows';
    END
    $$;
    \quit
\endif

\echo '========================================='
\echo 'TPC-H Query 19'
\echo '========================================='
\i sql/tpch/19.sql
SELECT CASE WHEN :ROW_COUNT = 1 THEN 'true' ELSE 'false' END AS q19_passed \gset
\if :q19_passed
    \echo 'Query 19: PASSED'
\else
    DROP EXTENSION pg_deeplake CASCADE;
    DO $$
    BEGIN
        RAISE EXCEPTION 'Query 19 FAILED: Expected 1 row';
    END
    $$;
    \quit
\endif

\echo '========================================='
\echo 'TPC-H Query 20'
\echo '========================================='
\i sql/tpch/20.sql
SELECT CASE WHEN :ROW_COUNT = 1 THEN 'true' ELSE 'false' END AS q20_passed \gset
\if :q20_passed
    \echo 'Query 20: PASSED'
\else
    DROP EXTENSION pg_deeplake CASCADE;
    DO $$
    BEGIN
        RAISE EXCEPTION 'Query 20 FAILED: Expected 1 row';
    END
    $$;
    \quit
\endif

\echo '========================================='
\echo 'TPC-H Query 21'
\echo '========================================='
\i sql/tpch/21.sql
SELECT CASE WHEN :ROW_COUNT = 10 THEN 'true' ELSE 'false' END AS q21_passed \gset
\if :q21_passed
    \echo 'Query 21: PASSED'
\else
    DROP EXTENSION pg_deeplake CASCADE;
    DO $$
    BEGIN
        RAISE EXCEPTION 'Query 21 FAILED: Expected 10 rows';
    END
    $$;
    \quit
\endif

\echo '========================================='
\echo 'TPC-H Query 22'
\echo '========================================='
\i sql/tpch/22.sql
SELECT CASE WHEN :ROW_COUNT = 7 THEN 'true' ELSE 'false' END AS q22_passed \gset
\if :q22_passed
    \echo 'Query 22: PASSED'
\else
    DROP EXTENSION pg_deeplake CASCADE;
    DO $$
    BEGIN
        RAISE EXCEPTION 'Query 22 FAILED: Expected 7 rows';
    END
    $$;
    \quit
\endif

RESET pg_deeplake.use_deeplake_executor;

\echo '========================================='
\echo 'All TPC-H queries completed'
\echo '========================================='
