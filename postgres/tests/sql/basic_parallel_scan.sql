\i sql/utils.psql

-- SET max_parallel_workers_per_gather = 4;
-- SET parallel_tuple_cost = 0.1;
-- SET parallel_setup_cost = 1000;

DROP TABLE IF EXISTS people CASCADE;
DROP EXTENSION IF EXISTS pg_deeplake CASCADE;
CREATE EXTENSION pg_deeplake;
SET pg_deeplake.use_custom_nodes = off;
CREATE TABLE people (name text, last_name text, age int) USING deeplake;

DO $$ BEGIN
    INSERT INTO people SELECT 'n'||i, 'l'||i, i FROM generate_series(1, 400000) i;
    PERFORM assert_table_row_count(400000, 'people');

    IF is_using_parallel_seq_scan('SELECT COUNT(*) FROM people;') THEN
        RAISE EXCEPTION 'Query must not use a parallel sequential scan by default!';
    END IF;

    SET pg_deeplake.enable_parallel_workers = true;

    IF NOT is_using_parallel_seq_scan('SELECT COUNT(*) FROM people;') THEN
        RAISE EXCEPTION 'Query must use a parallel sequential scan when enabled!';
    END IF;

    IF NOT is_using_parallel_seq_scan('SELECT COUNT(*), AVG(age) FROM people;') THEN
        RAISE EXCEPTION 'Query must use a parallel sequential scan with LIMIT!';
    END IF;

    RAISE NOTICE 'Test passed';
    EXCEPTION
    WHEN OTHERS THEN
        RAISE NOTICE 'ERROR: Test failed: %', SQLERRM;
END $$;

-- Cleanup
DROP TABLE IF EXISTS people CASCADE;
DROP EXTENSION IF EXISTS pg_deeplake CASCADE;
RESET pg_deeplake.enable_parallel_workers;