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

        PERFORM assert_table_row_count(4, 'vectors');

        -- Test index usage
        SET enable_seqscan = off;
        CREATE TEMP TABLE expected_vectors (id INTEGER, score REAL);
        INSERT INTO expected_vectors VALUES 
        (1, 1),
        (2, 0.97463185),
        (3, 0.959412),
        (4, 0.80178374);
        CREATE TEMP TABLE actual AS SELECT id, v1 <#> ARRAY[1.0, 2.0, 3.0] AS score FROM vectors ORDER BY score LIMIT 5;

        IF EXISTS (
            SELECT 1
            FROM expected_vectors e
            JOIN actual a USING (id, score)
            WHERE abs(e.score - a.score) > 1e-6
        ) THEN
            RAISE EXCEPTION 'Test failed: query result differs from expected';
        END IF;

        DROP TABLE actual;
        DROP TABLE expected_vectors;

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
