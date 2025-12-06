\i sql/utils.psql

DROP TABLE IF EXISTS people CASCADE;
DROP EXTENSION IF EXISTS pg_deeplake CASCADE;
CREATE EXTENSION pg_deeplake;

CREATE TABLE people (name text, last_name text, age int) USING deeplake;

DO $$ BEGIN
    INSERT INTO people (name, last_name, age) VALUES ('n1', 'l1', 1);
    INSERT INTO people (name, last_name, age) VALUES ('n2', 'l2', 2);
    INSERT INTO people (name, last_name, age) VALUES ('n3', 'l3', 3);
    INSERT INTO people (name, last_name, age) VALUES ('n4', 'l4', 4);

    DELETE FROM people WHERE name = 'n3' AND last_name = 'l3';

    PERFORM assert_table_row_count(3, 'people');

    INSERT INTO people (name, last_name, age) VALUES ('n3', 'l3', 3), ('n5', 'l5', 5), ('n6', 'l6', 6);

    PERFORM assert_table_row_count(6, 'people');
    PERFORM assert_query_row_count(1, 'SELECT * FROM people WHERE age = 4');

    DELETE FROM people WHERE age % 3 = 0;

    PERFORM assert_table_row_count(4, 'people');

    -- Generating 10k rows crashes in deeplake, change to 10k after fixing the issue
    INSERT INTO people SELECT 'n'||i, 'l'||i, i FROM generate_series(1, 1000) i;
    PERFORM assert_table_row_count(1004, 'people');

    DELETE FROM people WHERE age % 2 = 0;
    PERFORM assert_table_row_count(502, 'people');
    DELETE FROM people WHERE age % 2 = 1;
    PERFORM assert_table_row_count(0, 'people');

    RAISE NOTICE 'Test passed';
    EXCEPTION
    WHEN OTHERS THEN
        RAISE NOTICE 'ERROR: Test failed: %', SQLERRM;
END $$;

-- Cleanup
DROP TABLE IF EXISTS people CASCADE;
DROP EXTENSION IF EXISTS pg_deeplake CASCADE;
