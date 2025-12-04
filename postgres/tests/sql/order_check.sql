\i sql/utils.psql

-- ALTER TABLE vectors DROP COLUMN v1;
DROP TABLE IF EXISTS vectors CASCADE;
DROP EXTENSION IF EXISTS pg_deeplake CASCADE;
CREATE EXTENSION pg_deeplake;

DO $$ BEGIN
    BEGIN
        CREATE TABLE vectors (
            id SERIAL PRIMARY KEY,
            v1 float4[],
            v2 float4[]
        ) USING deeplake WITH (dataset_path = 'mem://');
        CREATE INDEX desc_index ON vectors USING deeplake_index (v2 DESC);

        -- Verify index exists
        PERFORM assert_query_row_count(1, 'SELECT * FROM pg_class WHERE relname = ''desc_index''');

        INSERT INTO vectors (v1, v2) VALUES (ARRAY[1.0, 2.0, 3.0], ARRAY[1.0, 2.0, 3.0]);
        INSERT INTO vectors (v1, v2) VALUES (ARRAY[4.0, 5.0, 6.0], ARRAY[4.0, 5.0, 6.0]);
        INSERT INTO vectors (v1, v2) VALUES (ARRAY[7.0, 8.0, 9.0], ARRAY[7.0, 8.0, 9.0]);
        INSERT INTO vectors (v1, v2) VALUES (ARRAY[10.0, 11.0, 12.0], ARRAY[10.0, 11.0, 12.0]);

        -- Test index usage
        SET enable_seqscan = off;
        CREATE TEMP TABLE expected_vectors_1 (id INTEGER, v1 REAL[], v2 REAL[]);
        INSERT INTO expected_vectors_1 VALUES 
        (2, '{4,5,6}', '{4,5,6}'),
        (3, '{7,8,9}', '{7,8,9}'),
        (4, '{10,11,12}', '{10,11,12}'),
        (1, '{1,2,3}', '{1,2,3}');
        PERFORM assert_query_result('SELECT * FROM vectors ORDER BY v1 <#> ARRAY[4.1, 5.2, 6.3] DESC LIMIT 5', 'expected_vectors_1');

        CREATE TEMP TABLE expected_vectors_2 (id INTEGER, v1 REAL[], v2 REAL[]);
        INSERT INTO expected_vectors_2 VALUES 
        (2, '{4,5,6}', '{4,5,6}'),
        (3, '{7,8,9}', '{7,8,9}'),
        (4, '{10,11,12}', '{10,11,12}'),
        (1, '{1,2,3}', '{1,2,3}');
        PERFORM assert_query_result('SELECT * FROM vectors ORDER BY v2 <#> ARRAY[4.0, 5.0, 6.0] LIMIT 5', 'expected_vectors_2');

        RAISE NOTICE 'Test passed';
        EXCEPTION
        WHEN OTHERS THEN
            RAISE NOTICE 'ERROR: Test failed: %', SQLERRM;
    END;

    -- Cleanup
    DROP INDEX IF EXISTS asc_index CASCADE;
    DROP INDEX IF EXISTS desc_index CASCADE;
    DROP TABLE IF EXISTS vectors CASCADE;
    DROP EXTENSION IF EXISTS pg_deeplake CASCADE;
    RESET enable_seqscan;
    RESET log_min_messages;
    RESET client_min_messages;
END;
$$;
