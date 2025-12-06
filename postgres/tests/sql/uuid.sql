\i sql/utils.psql

DROP TABLE IF EXISTS people CASCADE;
DROP EXTENSION IF EXISTS pg_deeplake CASCADE;
CREATE EXTENSION pg_deeplake;

DO $$ BEGIN
    BEGIN
        CREATE TABLE people (
            id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            name text,
            last_name text,
            age int
        ) USING deeplake;

        INSERT INTO people(name, last_name, age) SELECT 'name' || i, 'last' || i, i FROM generate_series(1, 5) AS s(i);
        INSERT INTO people(id, name, last_name, age) VALUES ('550e8400-e29b-41d4-a716-446655440000'::uuid, 'John', 'Doe', 30);

        PERFORM assert_query_row_count(6, 'SELECT * FROM people');

        PERFORM assert_query_row_count(1, 'SELECT * FROM people WHERE id = ''550e8400-e29b-41d4-a716-446655440000''');

        PERFORM assert_query_row_count(1, 'SELECT COUNT(*) AS total_rows, COUNT(DISTINCT id) AS unique_ids FROM people');

        RAISE NOTICE 'Test passed';
        EXCEPTION
        WHEN OTHERS THEN
            RAISE NOTICE 'ERROR: Test failed: %', SQLERRM;
    END;

    -- Cleanup
    DROP TABLE IF EXISTS people CASCADE;
    DROP EXTENSION IF EXISTS pg_deeplake CASCADE;
END;
$$;
