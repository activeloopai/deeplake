\i sql/utils.psql

DROP TABLE IF EXISTS people CASCADE;
DROP EXTENSION IF EXISTS pg_deeplake CASCADE;
CREATE EXTENSION pg_deeplake;
SET pg_deeplake.use_deeplake_executor = off;

CREATE TABLE people (name text, last_name text, age int) USING deeplake;

DO $$ BEGIN
    INSERT INTO people (name, last_name, age) VALUES ('n1', 'l1', 1);
    INSERT INTO people (name, last_name, age) VALUES ('n2', 'l2', 2);
    INSERT INTO people (name, last_name, age) VALUES ('n3', 'l3', 3);
    INSERT INTO people (name, last_name, age) VALUES ('n4', 'l4', 4);

    PERFORM assert_table_row_count(4, 'people');

    INSERT INTO people (name, last_name, age) VALUES ('n4', 'l4', 4), ('n5', 'l5', 5), ('n6', 'l6', 6);

    PERFORM assert_table_row_count(7, 'people');
    PERFORM assert_query_row_count(2, 'SELECT * FROM people WHERE age = 4');

    INSERT INTO people SELECT 'n'||i, 'l'||i, i FROM generate_series(1, 100000) i;
    INSERT INTO people SELECT 'n'||i, 'l'||i, i FROM generate_series(1, 100000) i;
    PERFORM assert_table_row_count(200007, 'people');

    IF is_using_index_scan('SELECT ctid, age FROM people WHERE age = 4;') THEN
        RAISE EXCEPTION 'Query should not use an index scan!';
    END IF;

    CREATE INDEX idx_people_age ON people(age);

    IF NOT is_using_index_scan('SELECT ctid, age FROM people WHERE age = 4;') THEN
        RAISE EXCEPTION 'Query should use an index scan!';
    END IF;

    PERFORM assert_query_row_count(4, 'SELECT ctid, age FROM people WHERE age = 4');

    DROP INDEX idx_people_age;

    RAISE NOTICE 'Test passed';
    EXCEPTION
    WHEN OTHERS THEN
        RAISE NOTICE 'ERROR: Test failed: %', SQLERRM;
END $$;

-- Cleanup
DROP TABLE IF EXISTS people CASCADE;
DROP EXTENSION IF EXISTS pg_deeplake CASCADE;
RESET pg_deeplake.use_deeplake_executor;
