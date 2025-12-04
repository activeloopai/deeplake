\i sql/utils.psql

DROP TABLE IF EXISTS people CASCADE;
DROP EXTENSION IF EXISTS pg_deeplake CASCADE;
CREATE EXTENSION pg_deeplake;
SET pg_deeplake.use_deeplake_executor = off;
CREATE TABLE people (
    id SERIAL PRIMARY KEY,
    embedding EMBEDDING,
    name VARCHAR(50),
    last_name VARCHAR(50),
    age INT
) USING deeplake;

DO $$ BEGIN
    DECLARE
        selected_ctid tid;
        original_ctid tid;
        result_ctid tid;
        pass_count INTEGER := 0;
        fail_count INTEGER := 0;
        i INTEGER;
        knn_query TEXT;
    BEGIN
        CREATE INDEX index_for_emb ON people USING deeplake_index (embedding DESC);-- WITH (dataset_path = 'path');
        PERFORM assert_query_row_count(1, 'SELECT * FROM pg_class WHERE relname = ''index_for_emb''');

        -- Insert 100 rows into the table
        INSERT INTO people (embedding, name, last_name, age)
        SELECT
            generate_random_float_array(1024),                      -- Random float array with 1024 elements
            'Name_' || trunc(random() * 1000000),                   -- Random name
            'LastName_' || trunc(random() * 1000000),               -- Random last name
            trunc(random() * 100) + 1                               -- Random age between 1 and 100
        FROM generate_series(1, 2000);

        -- Verify that the table has 100 rows
        PERFORM assert_table_row_count(2000, 'people');

        SET enable_seqscan = off;
        FOR i IN 1..2000 LOOP
            -- Get the original CTID
            SELECT ctid INTO selected_ctid FROM people ORDER BY random() LIMIT 1;
            original_ctid := selected_ctid;

            knn_query := $q$
                WITH fixed_array AS (
                    SELECT embedding AS arr FROM people WHERE ctid = '(100,1)'
                )
                SELECT ctid
                FROM people, fixed_array
                ORDER BY embedding <#> (SELECT arr FROM fixed_array LIMIT 1)
                LIMIT 1;
            $q$;

            IF NOT is_using_index_scan(knn_query) THEN
                RAISE EXCEPTION 'Query must use an index scan!';
            END IF;

            -- Perform the KNN search
            WITH fixed_array AS (
                SELECT embedding AS arr FROM people WHERE ctid = selected_ctid
            )
            SELECT ctid INTO result_ctid
            FROM people, fixed_array
            ORDER BY embedding <#> (SELECT arr FROM fixed_array LIMIT 1)
            LIMIT 1;

            -- Compare results
            IF original_ctid IS DISTINCT FROM result_ctid THEN
                fail_count := fail_count + 1;
                RAISE NOTICE 'Run %: ❌ Mismatch - Original CTID: %, KNN result CTID: %', i, original_ctid, result_ctid;
            ELSE
                pass_count := pass_count + 1;
                RAISE NOTICE 'Run %: ✅ Match - Expected CTID: %', i, original_ctid;
            END IF;
        END LOOP;

        IF pass_count < 1998 THEN
            RAISE EXCEPTION 'Test failed! Only %/10 runs passed.', pass_count;
        ELSE
            RAISE NOTICE 'Test passed! %/10 runs were successful.', pass_count;
        END IF;

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
