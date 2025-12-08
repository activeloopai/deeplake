\i sql/utils.psql

DROP TABLE IF EXISTS json_test CASCADE;
DROP EXTENSION IF EXISTS pg_deeplake CASCADE;
CREATE EXTENSION pg_deeplake;

CREATE TABLE json_test (
    json_col json,
    jsonb_col jsonb
) USING deeplake;

DO $$ BEGIN
    -- Basic inserts with different JSON values
    INSERT INTO json_test (json_col, jsonb_col)
    VALUES
        ('{"name": "Alice", "age": 30, "city": "NYC"}', '{"name": "Alice", "age": 30, "city": "NYC"}'),
        ('{"name": "Bob", "age": 25, "city": "LA"}', '{"name": "Bob", "age": 25, "city": "LA"}'),
        ('{"name": "Charlie", "age": 35, "city": "SF"}', '{"name": "Charlie", "age": 35, "city": "SF"}'),
        ('{"name": "David", "age": 40, "city": "NYC"}', '{"name": "David", "age": 40, "city": "NYC"}');

    PERFORM assert_table_row_count(4, 'json_test');

    -- Additional inserts including duplicate for testing
    INSERT INTO json_test (json_col, jsonb_col)
    VALUES
        ('{"name": "David", "age": 40, "city": "NYC"}', '{"name": "David", "age": 40, "city": "NYC"}'),
        ('{"name": "Eve", "age": 28, "city": "Boston"}', '{"name": "Eve", "age": 28, "city": "Boston"}'),
        ('{"name": "Frank", "age": 32, "city": "Seattle"}', '{"name": "Frank", "age": 32, "city": "Seattle"}');

    -- Basic count assertions
    PERFORM assert_table_row_count(7, 'json_test');

    -- Test exact matches for JSON columns
    PERFORM assert_query_row_count(2, 'SELECT * FROM json_test WHERE json_col::text = ''{"name": "David", "age": 40, "city": "NYC"}''');
    PERFORM assert_query_row_count(1, 'SELECT * FROM json_test WHERE json_col::text = ''{"name": "Alice", "age": 30, "city": "NYC"}''');

    -- Test exact matches for JSONB columns (note: JSONB may reorder keys)
    EXECUTE 'SET LOCAL pg_deeplake.use_deeplake_executor = false';
     -- Disable deeplake executor to ensure consistent ordering for JSONB text comparison
    PERFORM assert_query_row_count(2, 'SELECT * FROM json_test WHERE jsonb_col = ''{"name": "David", "age": 40, "city": "NYC"}''::jsonb');
    PERFORM assert_query_row_count(1, 'SELECT * FROM json_test WHERE jsonb_col = ''{"name": "Bob", "age": 25, "city": "LA"}''::jsonb');

    -- Test JSON field extraction and filtering (using ->> for text extraction)
    PERFORM assert_query_row_count(1, 'SELECT * FROM json_test WHERE json_col->>''name'' = ''Alice''');
    PERFORM assert_query_row_count(2, 'SELECT * FROM json_test WHERE json_col->>''name'' = ''David''');
    PERFORM assert_query_row_count(3, 'SELECT * FROM json_test WHERE json_col->>''city'' = ''NYC''');
    PERFORM assert_query_row_count(1, 'SELECT * FROM json_test WHERE json_col->>''city'' = ''SF''');

    -- Test JSONB field extraction and filtering (using ->> for text extraction)
    PERFORM assert_query_row_count(1, 'SELECT * FROM json_test WHERE jsonb_col->>''name'' = ''Alice''');
    PERFORM assert_query_row_count(2, 'SELECT * FROM json_test WHERE jsonb_col->>''name'' = ''David''');
    PERFORM assert_query_row_count(3, 'SELECT * FROM json_test WHERE jsonb_col->>''city'' = ''NYC''');
    PERFORM assert_query_row_count(1, 'SELECT * FROM json_test WHERE jsonb_col->>''city'' = ''Boston''');

    -- Test numeric comparisons on JSON fields
    PERFORM assert_query_row_count(4, 'SELECT * FROM json_test WHERE (json_col->>''age'')::int > 30');
    PERFORM assert_query_row_count(5, 'SELECT * FROM json_test WHERE (json_col->>''age'')::int >= 30');
    PERFORM assert_query_row_count(2, 'SELECT * FROM json_test WHERE (json_col->>''age'')::int < 30');
    PERFORM assert_query_row_count(1, 'SELECT * FROM json_test WHERE (json_col->>''age'')::int = 25');

    -- Test numeric comparisons on JSONB fields
    PERFORM assert_query_row_count(4, 'SELECT * FROM json_test WHERE (jsonb_col->>''age'')::int > 30');
    PERFORM assert_query_row_count(5, 'SELECT * FROM json_test WHERE (jsonb_col->>''age'')::int >= 30');
    PERFORM assert_query_row_count(2, 'SELECT * FROM json_test WHERE (jsonb_col->>''age'')::int < 30');
    PERFORM assert_query_row_count(1, 'SELECT * FROM json_test WHERE (jsonb_col->>''age'')::int = 28');

    -- Test JSONB containment operator (@>)
    PERFORM assert_query_row_count(3, 'SELECT * FROM json_test WHERE jsonb_col @> ''{"city": "NYC"}''::jsonb');
    PERFORM assert_query_row_count(1, 'SELECT * FROM json_test WHERE jsonb_col @> ''{"name": "Alice", "age": 30}''::jsonb');
    PERFORM assert_query_row_count(2, 'SELECT * FROM json_test WHERE jsonb_col @> ''{"age": 40}''::jsonb');

    -- Test JSONB existence operator (?)
    PERFORM assert_query_row_count(7, 'SELECT * FROM json_test WHERE jsonb_col ? ''name''');
    PERFORM assert_query_row_count(7, 'SELECT * FROM json_test WHERE jsonb_col ? ''age''');
    PERFORM assert_query_row_count(7, 'SELECT * FROM json_test WHERE jsonb_col ? ''city''');
    PERFORM assert_query_row_count(0, 'SELECT * FROM json_test WHERE jsonb_col ? ''email''');

    -- Insert complex nested JSON structures
    INSERT INTO json_test (json_col, jsonb_col)
    VALUES
        ('{"person": {"name": "Grace", "details": {"age": 45, "hobbies": ["reading", "hiking"]}}}',
         '{"person": {"name": "Grace", "details": {"age": 45, "hobbies": ["reading", "hiking"]}}}'),
        ('{"person": {"name": "Henry", "details": {"age": 50, "hobbies": ["cooking", "gaming"]}}}',
         '{"person": {"name": "Henry", "details": {"age": 50, "hobbies": ["cooking", "gaming"]}}}');

    PERFORM assert_table_row_count(9, 'json_test');

    -- Test nested field extraction (use ->> for final text extraction)
    PERFORM assert_query_row_count(1, 'SELECT * FROM json_test WHERE json_col->''person''->>''name'' = ''Grace''');
    PERFORM assert_query_row_count(1, 'SELECT * FROM json_test WHERE jsonb_col->''person''->>''name'' = ''Henry''');
    PERFORM assert_query_row_count(1, 'SELECT * FROM json_test WHERE (json_col->''person''->''details''->>''age'')::int = 45');
    PERFORM assert_query_row_count(1, 'SELECT * FROM json_test WHERE (jsonb_col->''person''->''details''->>''age'')::int = 50');

    -- Test array containment in JSONB
    PERFORM assert_query_row_count(1, 'SELECT * FROM json_test WHERE jsonb_col->''person''->''details''->''hobbies'' @> ''["reading"]''::jsonb');
    PERFORM assert_query_row_count(1, 'SELECT * FROM json_test WHERE jsonb_col->''person''->''details''->''hobbies'' @> ''["gaming"]''::jsonb');

    -- Bulk insert with JSON generation
    INSERT INTO json_test
    SELECT
        json_build_object(
            'id', i,
            'name', 'User_' || i,
            'score', (i * 10) % 100,
            'active', (i % 2 = 0)
        ),
        jsonb_build_object(
            'id', i,
            'name', 'User_' || i,
            'score', (i * 10) % 100,
            'active', (i % 2 = 0)
        )
    FROM generate_series(1, 10000) i;

    PERFORM assert_table_row_count(10009, 'json_test');

    -- Test filtering on bulk inserted data
    PERFORM assert_query_row_count(5000, 'SELECT * FROM json_test WHERE (jsonb_col->>''active'')::boolean = true');
    PERFORM assert_query_row_count(5000, 'SELECT * FROM json_test WHERE (jsonb_col->>''active'')::boolean = false');
    PERFORM assert_query_row_count(1000, 'SELECT * FROM json_test WHERE (jsonb_col->>''score'')::int = 50');
    EXECUTE 'RESET pg_deeplake.use_deeplake_executor';

    -- Test null JSON and JSONB values
    INSERT INTO json_test (json_col, jsonb_col) VALUES (NULL, NULL);
    PERFORM assert_table_row_count(10010, 'json_test');
    PERFORM assert_query_row_count(1, 'SELECT * FROM json_test WHERE json_col IS NULL');
    PERFORM assert_query_row_count(1, 'SELECT * FROM json_test WHERE jsonb_col IS NULL');

    RAISE NOTICE 'Test passed';
    EXCEPTION
    WHEN OTHERS THEN
        RAISE NOTICE 'ERROR: Test failed: %', SQLERRM;
END $$;

-- Cleanup
DROP TABLE IF EXISTS json_test CASCADE;
DROP EXTENSION IF EXISTS pg_deeplake CASCADE;
RESET pg_deeplake.use_deeplake_executor;
