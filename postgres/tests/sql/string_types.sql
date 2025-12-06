\i sql/utils.psql

DROP TABLE IF EXISTS type_test CASCADE;
DROP EXTENSION IF EXISTS pg_deeplake CASCADE;
CREATE EXTENSION pg_deeplake;

CREATE TABLE type_test (
    char_col char,
    varchar_col varchar(50),
    varchar_1_col varchar(1),
    varchar_generic_col varchar,
    bpchar_col char(13),
    bpchar_1_col char(1),
    text_col text
) USING deeplake;

DO $$ BEGIN
    -- Basic inserts with different string values
    INSERT INTO type_test (char_col, varchar_col, varchar_1_col, varchar_generic_col, bpchar_col, bpchar_1_col, text_col)
    VALUES
        ('a', 'varchar_example_1', 'v', 'varchar_generic_example_1', 'bpchar_ex1', 'x', 'text_example_1'),
        ('b', 'varchar_example_2', 'w', 'varchar_generic_example_2', 'bpchar_ex2', 'y', 'text_example_2'),
        ('c', 'varchar_example_3', 'x', 'varchar_generic_example_3', 'bpchar_ex3', 'z', 'text_example_3'),
        ('d', 'varchar_example_4', 'y', 'varchar_generic_example_4', 'bpchar_ex4', 'a', 'text_example_4');

    PERFORM assert_table_row_count(4, 'type_test');

    -- Additional inserts including duplicate for testing
    INSERT INTO type_test (char_col, varchar_col, varchar_1_col, varchar_generic_col, bpchar_col, bpchar_1_col, text_col)
    VALUES
        ('d', 'varchar_example_4', 'y', 'varchar_generic_example_4', 'bpchar_ex4', 'a', 'text_example_4'),
        ('e', 'varchar_example_5', 'z', 'varchar_generic_example_5', 'bpchar_ex5', 'b', 'text_example_5'),
        ('f', 'varchar_example_6', 'v', 'varchar_generic_example_6', 'bpchar_ex6', 'c', 'text_example_6');

    -- Basic count assertions
    PERFORM assert_table_row_count(7, 'type_test');

    -- Test exact matches for each string column
    PERFORM assert_query_row_count(1, 'SELECT * FROM type_test WHERE char_col = ''a''');
    PERFORM assert_query_row_count(2, 'SELECT * FROM type_test WHERE char_col = ''d''');
    PERFORM assert_query_row_count(2, 'SELECT * FROM type_test WHERE varchar_col = ''varchar_example_4''');
    PERFORM assert_query_row_count(1, 'SELECT * FROM type_test WHERE varchar_col = ''varchar_example_1''');
    PERFORM assert_query_row_count(2, 'SELECT * FROM type_test WHERE varchar_1_col = ''v''');
    PERFORM assert_query_row_count(2, 'SELECT * FROM type_test WHERE varchar_1_col = ''y''');
    PERFORM assert_query_row_count(2, 'SELECT * FROM type_test WHERE varchar_generic_col = ''varchar_generic_example_4''');
    PERFORM assert_query_row_count(1, 'SELECT * FROM type_test WHERE varchar_generic_col = ''varchar_generic_example_5''');
    PERFORM assert_query_row_count(2, 'SELECT * FROM type_test WHERE bpchar_col = ''bpchar_ex4''');
    PERFORM assert_query_row_count(1, 'SELECT * FROM type_test WHERE bpchar_col = ''bpchar_ex6''');
    PERFORM assert_query_row_count(2, 'SELECT * FROM type_test WHERE bpchar_1_col = ''a''');
    PERFORM assert_query_row_count(1, 'SELECT * FROM type_test WHERE bpchar_1_col = ''c''');
    PERFORM assert_query_row_count(2, 'SELECT * FROM type_test WHERE text_col = ''text_example_4''');
    PERFORM assert_query_row_count(1, 'SELECT * FROM type_test WHERE text_col = ''text_example_2''');

    -- Test string comparison operators (lexicographic ordering)
    PERFORM assert_query_row_count(4, 'SELECT * FROM type_test WHERE char_col > ''c''');
    PERFORM assert_query_row_count(2, 'SELECT * FROM type_test WHERE char_col <= ''b''');
    PERFORM assert_query_row_count(4, 'SELECT * FROM type_test WHERE varchar_col >= ''varchar_example_4''');
    PERFORM assert_query_row_count(3, 'SELECT * FROM type_test WHERE varchar_col < ''varchar_example_4''');
    PERFORM assert_query_row_count(6, 'SELECT * FROM type_test WHERE text_col >= ''text_example_2''');

    -- Test LIKE pattern matching
    PERFORM assert_query_row_count(7, 'SELECT * FROM type_test WHERE varchar_col LIKE ''varchar_example_%''');
    PERFORM assert_query_row_count(1, 'SELECT * FROM type_test WHERE varchar_col LIKE ''varchar_example_1%''');
    PERFORM assert_query_row_count(1, 'SELECT * FROM type_test WHERE bpchar_col LIKE ''bpchar_ex1%''');
    PERFORM assert_query_row_count(7, 'SELECT * FROM type_test WHERE text_col LIKE ''%example_%''');

    -- Bulk insert with string generation
    INSERT INTO type_test
    SELECT
        chr(97 + (i % 26)),
        'varchar_bulk_' || i,
        chr(97 + (i % 10)),
        'generic_varchar_' || i,
        'bpchar_' || LPAD(i::text, 4, '0'),
        chr(97 + (i % 5)),
        'text_value_' || i
    FROM generate_series(1, 10000) i;

    PERFORM assert_table_row_count(10007, 'type_test');

    RAISE NOTICE 'Test passed';
    EXCEPTION
    WHEN OTHERS THEN
        RAISE NOTICE 'ERROR: Test failed: %', SQLERRM;
END $$;

-- Cleanup
DROP TABLE IF EXISTS type_test CASCADE;
DROP EXTENSION IF EXISTS pg_deeplake CASCADE;
