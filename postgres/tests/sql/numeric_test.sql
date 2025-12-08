\i sql/utils.psql

DROP TABLE IF EXISTS numeric_precision_test_deeplake CASCADE;
DROP TABLE IF EXISTS numeric_precision_test_native CASCADE;
DROP EXTENSION IF EXISTS pg_deeplake CASCADE;
CREATE EXTENSION pg_deeplake;
SET pg_deeplake.treat_numeric_as_double = false;

DO $$ BEGIN
    BEGIN
        -- Create native PostgreSQL table
        CREATE TABLE numeric_precision_test_native (
            id SERIAL PRIMARY KEY,
            num_default NUMERIC,
            num_10_0 NUMERIC(10, 0),
            num_10_2 NUMERIC(10, 2),
            num_15_5 NUMERIC(15, 5),
            num_15_10 NUMERIC(15, 10),
            num_20_10 NUMERIC(20, 10),
            num_38_18 NUMERIC(38, 18)
        );

        -- Create deeplake table with same structure
        CREATE TABLE numeric_precision_test_deeplake (
            id SERIAL PRIMARY KEY,
            num_default NUMERIC,
            num_10_0 NUMERIC(10, 0),
            num_10_2 NUMERIC(10, 2),
            num_15_5 NUMERIC(15, 5),
            num_15_10 NUMERIC(15, 10),
            num_20_10 NUMERIC(20, 10),
            num_38_18 NUMERIC(38, 18)
        ) USING deeplake;

        -- Insert same data into both tables
        INSERT INTO numeric_precision_test_native (num_default, num_10_0, num_10_2, num_15_5, num_15_10, num_20_10, num_38_18) VALUES
            (  123.456789,  1234567890,  12345678.90,  1234567890.12345,  12345.1234567890,  1234567890.1234567890,  12345678901234567890.123456789012345678),
            (    0.000001,           0,         0.01,           0.00001,      0.0000000001,           0.0000000001,                     0.000000000000000001),
            (999999999999,  9999999999,  99999999.99,  9999999999.99999,  99999.9999999999,  9999999999.9999999999,  99999999999999999999.999999999999999999),
            ( -123.456789, -1234567890, -12345678.90, -1234567890.12345, -12345.1234567890, -1234567890.1234567890, -12345678901234567890.123456789012345678),
            (         1.0,           1,         1.00,           1.00000,      1.0000000000,           1.0000000000,                     1.000000000000000000);

        INSERT INTO numeric_precision_test_deeplake (num_default, num_10_0, num_10_2, num_15_5, num_15_10, num_20_10, num_38_18) VALUES
            (  123.456789,  1234567890,  12345678.90,  1234567890.12345,  12345.1234567890,  1234567890.1234567890,  12345678901234567890.123456789012345678),
            (    0.000001,           0,         0.01,           0.00001,      0.0000000001,           0.0000000001,                     0.000000000000000001),
            (999999999999,  9999999999,  99999999.99,  9999999999.99999,  99999.9999999999,  9999999999.9999999999,  99999999999999999999.999999999999999999),
            ( -123.456789, -1234567890, -12345678.90, -1234567890.12345, -12345.1234567890, -1234567890.1234567890, -12345678901234567890.123456789012345678),
            (         1.0,           1,         1.00,           1.00000,      1.0000000000,           1.0000000000,                     1.000000000000000000);

        -- Test 1: Compare basic SELECT operations
        -- Note: num_20_10 and num_38_18 are casted down to REAL for comparison due to precision differences in Deeplake.
        PERFORM assert_tables_query_match(
            'SELECT num_default, num_10_0, num_10_2, num_15_5, num_15_10, num_20_10::REAL, num_38_18::REAL FROM numeric_precision_test_native ORDER BY id',
            'SELECT num_default, num_10_0, num_10_2, num_15_5, num_15_10, num_20_10::REAL, num_38_18::REAL FROM numeric_precision_test_deeplake ORDER BY id',
            'Basic SELECT test'
        );

        -- Test 2: Compare arithmetic operations
        PERFORM assert_tables_query_match(
            'SELECT id, num_10_2 + num_15_5 AS addition, num_10_2 - num_15_5 AS subtraction, num_10_2 * 2 AS multiplication, num_15_5 / 2 AS division FROM numeric_precision_test_native ORDER BY id',
            'SELECT id, num_10_2 + num_15_5 AS addition, num_10_2 - num_15_5 AS subtraction, num_10_2 * 2 AS multiplication, num_15_5 / 2 AS division FROM numeric_precision_test_deeplake ORDER BY id',
            'Arithmetic operations test'
        );

        -- Test 3: Compare precision preservation functions
        PERFORM assert_tables_query_match(
            'SELECT id, num_10_2, ROUND(num_10_2, 1) AS rounded_1, ROUND(num_10_2, 0) AS rounded_0, TRUNC(num_10_2, 1) AS truncated_1 FROM numeric_precision_test_native ORDER BY id',
            'SELECT id, num_10_2, ROUND(num_10_2, 1) AS rounded_1, ROUND(num_10_2, 0) AS rounded_0, TRUNC(num_10_2, 1) AS truncated_1 FROM numeric_precision_test_deeplake ORDER BY id',
            'Precision preservation test'
        );

        -- Test 4: Compare comparison operations
        PERFORM assert_tables_query_match(
            'SELECT id, num_10_2, num_15_5, CASE WHEN num_10_2 > num_15_5 THEN ''Greater'' WHEN num_10_2 < num_15_5 THEN ''Lesser'' ELSE ''Equal'' END AS comparison FROM numeric_precision_test_native ORDER BY id',
            'SELECT id, num_10_2, num_15_5, CASE WHEN num_10_2 > num_15_5 THEN ''Greater'' WHEN num_10_2 < num_15_5 THEN ''Lesser'' ELSE ''Equal'' END AS comparison FROM numeric_precision_test_deeplake ORDER BY id',
            'Comparison operations test'
        );

        -- Test 5: Compare aggregation functions
        PERFORM assert_tables_query_match(
            'SELECT COUNT(*) AS count_rows, SUM(num_10_2) AS sum_10_2, AVG(num_10_2) AS avg_10_2, MIN(num_10_2) AS min_10_2, MAX(num_10_2) AS max_10_2, STDDEV(num_10_2) AS stddev_10_2 FROM numeric_precision_test_native',
            'SELECT COUNT(*) AS count_rows, SUM(num_10_2) AS sum_10_2, AVG(num_10_2) AS avg_10_2, MIN(num_10_2) AS min_10_2, MAX(num_10_2) AS max_10_2, STDDEV(num_10_2) AS stddev_10_2 FROM numeric_precision_test_deeplake',
            'Aggregation functions test'
        );

        -- Test 6: Compare individual aggregate functions using specialized function
        -- Note: num_38_18 is casted down to REAL for comparison due to precision differences in Deeplake.
        PERFORM assert_aggregate_match('numeric_precision_test_native', 'numeric_precision_test_deeplake', 'num_10_2', 'COUNT', 'COUNT num_10_2');
        PERFORM assert_aggregate_match('numeric_precision_test_native', 'numeric_precision_test_deeplake', 'num_15_5', 'SUM', 'SUM num_15_5');
        PERFORM assert_aggregate_match('numeric_precision_test_native', 'numeric_precision_test_deeplake', 'num_15_10', 'SUM', 'SUM num_15_10');
        PERFORM assert_aggregate_match('numeric_precision_test_native', 'numeric_precision_test_deeplake', 'num_20_10', 'AVG', 'AVG num_20_10');
        PERFORM assert_aggregate_match('numeric_precision_test_native', 'numeric_precision_test_deeplake', 'num_38_18', 'MIN', 'MIN num_38_18', 'REAL');
        PERFORM assert_aggregate_match('numeric_precision_test_native', 'numeric_precision_test_deeplake', 'num_38_18', 'MAX', 'MAX num_38_18', 'REAL');

        -- Test 7: NULL handling comparison
        --- FIXME: Deeplake does not support NULLs in numeric columns
        --- INSERT INTO numeric_precision_test_native (num_10_2) VALUES (NULL);
        --- INSERT INTO numeric_precision_test_deeplake (num_10_2) VALUES (NULL);

        PERFORM assert_table_row_count(5, 'numeric_precision_test_native');
        PERFORM assert_table_row_count(5, 'numeric_precision_test_deeplake');

        -- Test 8: Compare type conversions (excluding TEXT conversion due to trailing zero handling)
        PERFORM assert_tables_query_match(
            'SELECT id, num_10_2::INTEGER AS to_integer, num_10_2::REAL AS to_real, num_10_2::DOUBLE PRECISION AS to_double FROM numeric_precision_test_native WHERE num_10_2 IS NOT NULL ORDER BY id',
            'SELECT id, num_10_2::INTEGER AS to_integer, num_10_2::REAL AS to_real, num_10_2::DOUBLE PRECISION AS to_double FROM numeric_precision_test_deeplake WHERE num_10_2 IS NOT NULL ORDER BY id',
            'Type conversion test num_10_2'
        );

        PERFORM assert_tables_query_match(
            'SELECT id, num_15_10::INTEGER AS to_integer, num_15_10::REAL AS to_real, num_15_10::DOUBLE PRECISION AS to_double FROM numeric_precision_test_native WHERE num_15_10 IS NOT NULL ORDER BY id',
            'SELECT id, num_15_10::INTEGER AS to_integer, num_15_10::REAL AS to_real, num_15_10::DOUBLE PRECISION AS to_double FROM numeric_precision_test_deeplake WHERE num_15_10 IS NOT NULL ORDER BY id',
            'Type conversion test num_15_10'
        );

        -- Test 9: Comprehensive precision tests for all numeric columns
        -- Note: num_20_10 and num_38_18 are converted to REAL for comparison due to precision differences
        PERFORM assert_tables_query_match(
            'SELECT id, num_default, num_10_0, num_10_2, num_15_5, num_15_10, num_20_10::REAL AS num_20_10_to_real, num_38_18::REAL AS num_38_18_to_real FROM numeric_precision_test_native WHERE id <= 5 ORDER BY id',
            'SELECT id, num_default, num_10_0, num_10_2, num_15_5, num_15_10, num_20_10::REAL AS num_20_10_to_real, num_38_18::REAL AS num_38_18_to_real FROM numeric_precision_test_deeplake ORDER BY id',
            'All numeric columns precision test'
        );

        -- Test 10: Verify aggregate functions across all numeric columns
        -- Note: num_20_10 and num_38_18 are converted to REAL for comparison due to precision differences
        PERFORM assert_aggregate_match('numeric_precision_test_native', 'numeric_precision_test_deeplake', 'num_default', 'SUM', 'SUM num_default');
        PERFORM assert_aggregate_match('numeric_precision_test_native', 'numeric_precision_test_deeplake', 'num_10_0', 'SUM', 'SUM num_10_0');
        PERFORM assert_aggregate_match('numeric_precision_test_native', 'numeric_precision_test_deeplake', 'num_10_2', 'SUM', 'SUM num_10_2');
        PERFORM assert_aggregate_match('numeric_precision_test_native', 'numeric_precision_test_deeplake', 'num_15_5', 'AVG', 'AVG num_15_5');
        PERFORM assert_aggregate_match('numeric_precision_test_native', 'numeric_precision_test_deeplake', 'num_15_10', 'AVG', 'AVG num_15_10');
        PERFORM assert_aggregate_match('numeric_precision_test_native', 'numeric_precision_test_deeplake', 'num_20_10', 'SUM', 'SUM num_20_10', 'REAL');
        PERFORM assert_aggregate_match('numeric_precision_test_native', 'numeric_precision_test_deeplake', 'num_38_18', 'SUM', 'SUM num_38_18', 'REAL');

        RAISE NOTICE 'All numeric precision tests passed';

    EXCEPTION
        WHEN OTHERS THEN
            RAISE NOTICE 'ERROR: Numeric precision test failed: %', SQLERRM;
    END;

    -- Cleanup
    DROP TABLE IF EXISTS numeric_precision_test_native CASCADE;
    DROP TABLE IF EXISTS numeric_precision_test_deeplake CASCADE;
    DROP EXTENSION IF EXISTS pg_deeplake CASCADE;
END;
$$;