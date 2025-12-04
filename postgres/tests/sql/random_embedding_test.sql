\i sql/utils.psql

DROP TABLE IF EXISTS people CASCADE;
DROP EXTENSION IF EXISTS pg_deeplake CASCADE;
CREATE EXTENSION pg_deeplake;
SET pg_deeplake.use_deeplake_executor = off;

CREATE TABLE people (
    id SERIAL PRIMARY KEY,
    embedding FLOAT4[],
    name VARCHAR(50),
    last_name VARCHAR(50),
    age INT
) USING deeplake;

DO $$ BEGIN
    BEGIN
        CREATE INDEX index_for_emb ON people USING deeplake_index (embedding ASC);
        PERFORM assert_query_row_count(1, 'SELECT * FROM pg_class WHERE relname = ''index_for_emb''');

        -- Insert 100 rows into the table
        INSERT INTO people (embedding, name, last_name, age)
        SELECT
            generate_random_float_array(1024),                      -- Random float array with 1024 elements
            'Name_' || trunc(random() * 1000000),                   -- Random name
            'LastName_' || trunc(random() * 1000000),               -- Random last name
            trunc(random() * 100) + 1                               -- Random age between 1 and 100
        FROM generate_series(1, 100);

        -- Just to run the query and check if it works
        PERFORM execute_query('SELECT ctid FROM people ORDER BY embedding <#> generate_random_float_array(1024) LIMIT 1;');

        -- Check that first row is returned
        PERFORM execute_and_check_query_text('SELECT ctid FROM people LIMIT 1;', '(0,1)');

        SET enable_seqscan = off;
        -- With generated (non-fixed) array non-indexed scan is used
        IF is_using_index_scan('SELECT ctid FROM people ORDER BY embedding <#> generate_random_float_array(1024) LIMIT 1;') THEN
            RAISE EXCEPTION 'Query must use an index scan!';
        END IF;

        -- With fixed array index scan should be used
        IF NOT is_using_index_scan(
            'WITH fixed_array AS ('
            '    SELECT generate_random_float_array(1024) AS arr'
            ')'
            'SELECT ctid '
            'FROM people, fixed_array '
            'ORDER BY embedding <#> (SELECT arr FROM fixed_array LIMIT 1) '
            'LIMIT 1;') THEN
            RAISE EXCEPTION 'Query must use an index scan!';
        END IF;

        -- WITH rows_to_delete AS (
            --     SELECT id
            --     FROM people
            --     ORDER BY id
            --     LIMIT 10
            -- )
        -- DELETE FROM people
        -- WHERE id IN (SELECT id FROM rows_to_delete);
        EXCEPTION
        WHEN OTHERS THEN
            RAISE NOTICE 'ERROR: Test failed: %', SQLERRM;
    END;

    -- Cleanup
    DROP INDEX IF EXISTS index_for_emb CASCADE;
    DROP TABLE IF EXISTS people CASCADE;
    DROP EXTENSION IF EXISTS pg_deeplake CASCADE;
    RESET enable_seqscan;
    RESET log_min_messages;
    RESET client_min_messages;
    RESET pg_deeplake.use_deeplake_executor;
END;
$$;
