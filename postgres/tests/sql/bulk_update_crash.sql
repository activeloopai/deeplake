\i sql/utils.psql

DROP TABLE IF EXISTS people CASCADE;
DROP EXTENSION IF EXISTS pg_deeplake CASCADE;
CREATE EXTENSION pg_deeplake;

CREATE TABLE people (name text, last_name text, age int) USING deeplake;

DO $$ BEGIN
    -- Insert initial test data
    INSERT INTO people (name, last_name, age) VALUES ('n1', 'l1', 1);
    INSERT INTO people (name, last_name, age) VALUES ('n2', 'l2', 2);
    INSERT INTO people (name, last_name, age) VALUES ('n3', 'l3', 3);
    INSERT INTO people (name, last_name, age) VALUES ('n4', 'l4', 4);

    PERFORM assert_table_row_count(4, 'people');

    -- Test single row update
    UPDATE people SET age = 25 WHERE name = 'n1';
    PERFORM assert_query_row_count(1, 'SELECT * FROM people WHERE age = 25');

    -- Test multiple row update
    UPDATE people SET last_name = 'updated' WHERE age <= 4;
    PERFORM assert_query_row_count(3, 'SELECT * FROM people WHERE last_name = ''updated''');

    -- Test bulk update with condition
    UPDATE people SET age = age + 10 WHERE name LIKE 'n%';
    PERFORM assert_query_row_count(0, 'SELECT * FROM people WHERE age < 10');
END $$;

-- Add more test data via COPY
COPY people (name, last_name, age) FROM STDIN WITH CSV;
n5,l5,5
n6,l6,6
n7,l7,7
\.

DO $$ BEGIN
    PERFORM assert_table_row_count(7, 'people');

    -- Test update on COPY-inserted data
    UPDATE people SET name = 'updated_' || name WHERE age <= 7;
    PERFORM assert_query_row_count(3, 'SELECT * FROM people WHERE last_name LIKE ''updated%''');

    -- Test bulk update with JOIN-like pattern
    UPDATE people SET age = CASE 
        WHEN age > 15 THEN age - 5
        ELSE age + 5
    END;

    PERFORM assert_table_row_count(7, 'people');

    RAISE NOTICE 'Update test passed';
    EXCEPTION
    WHEN OTHERS THEN
        RAISE NOTICE 'ERROR: Update test failed: %', SQLERRM;
END $$;

-- Cleanup
DROP TABLE IF EXISTS people CASCADE;
DROP EXTENSION IF EXISTS pg_deeplake CASCADE;
