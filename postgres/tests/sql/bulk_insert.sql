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

    PERFORM assert_table_row_count(4, 'people');

    INSERT INTO people (name, last_name, age) VALUES ('n4', 'l4', 4), ('n5', 'l5', 5), ('n6', 'l6', 6);

    PERFORM assert_table_row_count(7, 'people');
    PERFORM assert_query_row_count(2, 'SELECT * FROM people WHERE age = 4');
END $$;

-- COPY outside the DO block
COPY people (name, last_name, age) FROM STDIN WITH CSV;
n7,l7,7
n8,l8,8
n9,l9,9
\.

DO $$ BEGIN
    PERFORM assert_table_row_count(10, 'people');

    INSERT INTO people SELECT 'n'||i, 'l'||i, i FROM generate_series(1, 10000) i;
    PERFORM assert_table_row_count(10010, 'people');

    RAISE NOTICE 'Test passed';
    EXCEPTION
    WHEN OTHERS THEN
        RAISE NOTICE 'ERROR: Test failed: %', SQLERRM;
END $$;

-- Cleanup
DROP TABLE IF EXISTS people CASCADE;
DROP EXTENSION IF EXISTS pg_deeplake CASCADE;
