\i sql/utils.psql

DROP TABLE IF EXISTS vectors CASCADE;
DROP EXTENSION IF EXISTS pg_deeplake CASCADE;
CREATE EXTENSION pg_deeplake;

DO $$ BEGIN
    DECLARE
        ds_path CONSTANT TEXT := 'current_dataset/';
        create_index_query TEXT;
    BEGIN
        CREATE TABLE vectors (
            id SERIAL PRIMARY KEY,
            v1 float4[],
            v2 float4[]
        ) USING deeplake WITH (dataset_path = 'current_dataset/');
        CREATE INDEX index_for_v1 ON vectors USING deeplake_index (v1 DESC);

        -- Verify index exists
        PERFORM assert_query_row_count(1, 'SELECT * FROM pg_class WHERE relname = ''index_for_v1''');
        PERFORM execute_and_check_query_text('SELECT index_name FROM pg_deeplake_metadata;', 'index_for_v1');

        IF NOT directory_exists(ds_path) OR directory_empty(ds_path) THEN
            RAISE EXCEPTION 'Error: Dataset directory "%" does not exists!', ds_path;
        END IF;

        EXCEPTION
        WHEN OTHERS THEN
            RAISE NOTICE 'ERROR: Test failed: %', SQLERRM;
    END;
END;
$$;

-- reconnect
\c

DO $$ BEGIN
    DECLARE
        ds_path CONSTANT TEXT := 'current_dataset/';
    BEGIN
        DROP INDEX index_for_v1 CASCADE;
        -- Verify that index dropped and dataset deleted
        PERFORM assert_query_row_count(0, 'SELECT * FROM pg_class WHERE relname = ''index_for_v1''');
        PERFORM execute_and_check_query_text('SELECT index_name FROM pg_deeplake_metadata;', '');

        IF directory_empty(ds_path) THEN
            RAISE EXCEPTION 'Error: Dataset directory "%" cleaned up after reconnect and DROP INDEX!', ds_path;
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
