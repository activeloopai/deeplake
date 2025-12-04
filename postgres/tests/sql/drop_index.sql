\i sql/utils.psql

DROP TABLE IF EXISTS vectors CASCADE;
DROP EXTENSION IF EXISTS pg_deeplake CASCADE;
CREATE EXTENSION pg_deeplake;

DO $$ BEGIN
    DECLARE
        dataset_path TEXT;
    BEGIN
        CREATE TABLE vectors (
            id SERIAL PRIMARY KEY,
            v1 float4[],
            v2 float4[]
        ) USING deeplake;
        CREATE INDEX index_for_v1 ON vectors USING deeplake_index (v1 DESC);

        -- Verify index exists
        PERFORM assert_query_row_count(1, 'SELECT * FROM pg_class WHERE relname = ''index_for_v1''');
        PERFORM execute_and_check_query_text('SELECT index_name FROM pg_deeplake_metadata;', 'index_for_v1');

        dataset_path := get_query_result('SELECT ds_path FROM pg_deeplake_tables WHERE table_name = (SELECT table_name FROM pg_deeplake_metadata LIMIT 1)');
        IF NOT directory_exists(dataset_path) THEN
            RAISE EXCEPTION 'Error: Dataset directory "%" does not exists!', dataset_path;
        END IF;

        DROP INDEX index_for_v1 CASCADE;

        -- Verify that index dropped and dataset deleted
        PERFORM assert_query_row_count(0, 'SELECT * FROM pg_class WHERE relname = ''index_for_v1''');
        PERFORM execute_and_check_query_text('SELECT index_name FROM pg_deeplake_metadata;', '');

        IF NOT directory_exists(dataset_path) THEN
            RAISE EXCEPTION 'Error: Dataset directory "%" should exist after DROP INDEX!', dataset_path;
        END IF;

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
