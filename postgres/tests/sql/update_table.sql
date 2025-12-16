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
        ) USING deeplake;
        CREATE INDEX index_for_v1 ON vectors USING deeplake_index (v1 DESC);

        -- Verify index exists
        PERFORM assert_query_row_count(1, 'SELECT * FROM pg_class WHERE relname = ''index_for_v1''');

        INSERT INTO vectors (v1, v2) VALUES (ARRAY[1.0, 2.0, 3.0], ARRAY[1.0, 2.0, 3.0]);
        INSERT INTO vectors (v1, v2) VALUES (ARRAY[4.0, 5.0, 6.0], ARRAY[4.0, 5.0, 6.0]);
        INSERT INTO vectors (v1, v2) VALUES (ARRAY[7.0, 8.0, 9.0], ARRAY[7.0, 8.0, 9.0]);
        INSERT INTO vectors (v1, v2) VALUES (ARRAY[0.0, 0.0, 444], ARRAY[0.0, 0.0, 444]);

        -- Test index usage
        SET enable_seqscan = off;
        CREATE TEMP TABLE expected_vectors (id INTEGER, v1 REAL[], v2 REAL[]);
        INSERT INTO expected_vectors VALUES
        (1, '{1,2,3}', '{1,2,3}'),
        (2, '{4,5,6}', '{4,5,6}'),
        (3, '{7,8,9}', '{7,8,9}'),
        (4, '{0,0,444}', '{0,0,444}');
        PERFORM assert_query_result('SELECT * FROM vectors ORDER BY v1 <#> ARRAY[1.0, 2.0, 3.0] LIMIT 5', 'expected_vectors');

        UPDATE vectors SET v1 = ARRAY[9.0, 10.0, 11.0] WHERE id = 2;
        DROP TABLE expected_vectors;
        CREATE TEMP TABLE expected_vectors (id INTEGER, v1 REAL[], v2 REAL[]);
        INSERT INTO expected_vectors VALUES
        (1, '{1,2,3}', '{1,2,3}'),
        (2, '{9,10,11}', '{4,5,6}'),
        (3, '{7,8,9}', '{7,8,9}'),
        (4, '{0,0,444}', '{0,0,444}');
        PERFORM assert_query_result('SELECT * FROM vectors ORDER BY v1 <#> ARRAY[1.0, 2.0, 3.0] LIMIT 5', 'expected_vectors');

        RAISE NOTICE 'Test passed';
        EXCEPTION
        WHEN OTHERS THEN
            RAISE NOTICE 'ERROR: Test failed: %', SQLERRM;
    END;

    -- Cleanup
    DROP INDEX IF EXISTS index_for_v1 CASCADE;
    DROP TABLE IF EXISTS vectors CASCADE;
    DROP EXTENSION IF EXISTS pg_deeplake CASCADE;
    RESET enable_seqscan;
    RESET log_min_messages;
    RESET client_min_messages;
END;
$$;
