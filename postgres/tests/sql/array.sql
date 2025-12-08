\i sql/utils.psql

DROP TABLE IF EXISTS array_test CASCADE;
DROP EXTENSION IF EXISTS pg_deeplake CASCADE;
CREATE EXTENSION pg_deeplake;
--SET pg_deeplake.use_deeplake_executor = false;

CREATE TABLE array_test (
    float4_array_1d EMBEDDING,
    float4_array_2d EMBEDDING_2D,
    bytea_array_1d bytea[],
    text_array_1d text[],
    varchar_array_1d varchar[]
) USING deeplake;

DO $$ BEGIN
    -- Basic inserts with different array values
    INSERT INTO array_test (float4_array_1d, float4_array_2d, bytea_array_1d, text_array_1d, varchar_array_1d)
    VALUES
        (ARRAY[1.0, 2.0, 3.0]::float4[], ARRAY[[1.0, 2.0], [3.0, 4.0]]::float4[][], ARRAY['\x01'::bytea, '\x02'::bytea, '\x03'::bytea], ARRAY['text1'::text, 'text2'::text, 'text3'::text], ARRAY['varchar1'::varchar, 'varchar2'::varchar, 'varchar3'::varchar]),
        (ARRAY[4.0, 5.0, 6.0]::float4[], ARRAY[[5.0, 6.0], [7.0, 8.0]]::float4[][], ARRAY['\x05'::bytea, '\x06'::bytea, '\x07'::bytea], ARRAY['text4'::text, 'text5'::text, 'text6'::text], ARRAY['varchar4'::varchar, 'varchar5'::varchar, 'varchar6'::varchar]),
        (ARRAY[7.0, 8.0, 9.0]::float4[], ARRAY[[9.0, 10.0], [11.0, 12.0]]::float4[][], ARRAY['\x09'::bytea, '\x0A'::bytea, '\x0B'::bytea], ARRAY['text7'::text, 'text8'::text, 'text9'::text], ARRAY['varchar7'::varchar, 'varchar8'::varchar, 'varchar9'::varchar]),
        (ARRAY[10.0, 11.0, 12.0]::float4[], ARRAY[[13.0, 14.0], [15.0, 16.0]]::float4[][], ARRAY['\x0D'::bytea, '\x0E'::bytea, '\x0F'::bytea], ARRAY['text10'::text, 'text11'::text, 'text12'::text], ARRAY['varchar10'::varchar, 'varchar11'::varchar, 'varchar12'::varchar]);

    PERFORM assert_table_row_count(4, 'array_test');

    -- Additional inserts including duplicate for testing
    INSERT INTO array_test (float4_array_1d, float4_array_2d, bytea_array_1d, text_array_1d, varchar_array_1d)
    VALUES
        (ARRAY[10.0, 11.0, 12.0]::float4[], ARRAY[[13.0, 14.0], [15.0, 16.0]]::float4[][], ARRAY['\x0D'::bytea, '\x0E'::bytea, '\x0F'::bytea], ARRAY['text13'::text, 'text14'::text, 'text15'::text], ARRAY['varchar13'::varchar, 'varchar14'::varchar, 'varchar15'::varchar]),
        (ARRAY[13.0, 14.0, 15.0]::float4[], ARRAY[[17.0, 18.0], [19.0, 20.0]]::float4[][], ARRAY['\x11'::bytea, '\x12'::bytea, '\x13'::bytea], ARRAY['text16'::text, 'text17'::text, 'text18'::text], ARRAY['varchar16'::varchar, 'varchar17'::varchar, 'varchar18'::varchar]),
        (ARRAY[16.0, 17.0, 18.0]::float4[], ARRAY[[21.0, 22.0], [23.0, 24.0]]::float4[][], ARRAY['\x15'::bytea, '\x16'::bytea, '\x17'::bytea], ARRAY['text19'::text, 'text20'::text, 'text21'::text], ARRAY['varchar19'::varchar, 'varchar20'::varchar, 'varchar21'::varchar]);

    -- Basic count assertions
    PERFORM assert_table_row_count(7, 'array_test');

    -- Test exact matches for float4 1D arrays
    PERFORM assert_query_row_count(1, 'SELECT * FROM array_test WHERE float4_array_1d = ARRAY[1.0, 2.0, 3.0]::float4[]');
    PERFORM assert_query_row_count(2, 'SELECT * FROM array_test WHERE float4_array_1d = ARRAY[10.0, 11.0, 12.0]::float4[]');
    PERFORM assert_query_row_count(1, 'SELECT * FROM array_test WHERE float4_array_1d = ARRAY[7.0, 8.0, 9.0]::float4[]');

    -- Test exact matches for float4 2D arrays
    PERFORM assert_query_row_count(1, 'SELECT * FROM array_test WHERE float4_array_2d = ARRAY[[1.0, 2.0], [3.0, 4.0]]::float4[][]');
    PERFORM assert_query_row_count(2, 'SELECT * FROM array_test WHERE float4_array_2d = ARRAY[[13.0, 14.0], [15.0, 16.0]]::float4[][]');
    PERFORM assert_query_row_count(1, 'SELECT * FROM array_test WHERE float4_array_2d = ARRAY[[9.0, 10.0], [11.0, 12.0]]::float4[][]');

    -- Test exact matches for bytea 1D arrays
    PERFORM assert_query_row_count(1, 'SELECT * FROM array_test WHERE bytea_array_1d = ARRAY[''\x01''::bytea, ''\x02''::bytea, ''\x03''::bytea]');
    PERFORM assert_query_row_count(2, 'SELECT * FROM array_test WHERE bytea_array_1d = ARRAY[''\x0D''::bytea, ''\x0E''::bytea, ''\x0F''::bytea]');
    PERFORM assert_query_row_count(1, 'SELECT * FROM array_test WHERE bytea_array_1d = ARRAY[''\x05''::bytea, ''\x06''::bytea, ''\x07''::bytea]');

    -- Test array element access for float4 1D arrays
    PERFORM assert_query_row_count(1, 'SELECT * FROM array_test WHERE float4_array_1d[1] = 1.0');
    PERFORM assert_query_row_count(1, 'SELECT * FROM array_test WHERE float4_array_1d[2] = 5.0');
    PERFORM assert_query_row_count(1, 'SELECT * FROM array_test WHERE float4_array_1d[3] = 9.0');
    PERFORM assert_query_row_count(2, 'SELECT * FROM array_test WHERE float4_array_1d[1] = 10.0');

    -- Test array element access for float4 2D arrays
    PERFORM assert_query_row_count(1, 'SELECT * FROM array_test WHERE float4_array_2d[1][1] = 1.0');
    PERFORM assert_query_row_count(1, 'SELECT * FROM array_test WHERE float4_array_2d[2][1] = 7.0');
    PERFORM assert_query_row_count(2, 'SELECT * FROM array_test WHERE float4_array_2d[1][1] = 13.0');

    -- Test array element access for bytea 1D arrays
    PERFORM assert_query_row_count(1, 'SELECT * FROM array_test WHERE bytea_array_1d[1] = ''\x01''::bytea');
    PERFORM assert_query_row_count(1, 'SELECT * FROM array_test WHERE bytea_array_1d[2] = ''\x06''::bytea');
    PERFORM assert_query_row_count(2, 'SELECT * FROM array_test WHERE bytea_array_1d[1] = ''\x0D''::bytea');

    -- Test array containment operator (@>) for float4 1D arrays
    PERFORM assert_query_row_count(1, 'SELECT * FROM array_test WHERE float4_array_1d @> ARRAY[1.0, 2.0]::float4[]');
    PERFORM assert_query_row_count(2, 'SELECT * FROM array_test WHERE float4_array_1d @> ARRAY[10.0]::float4[]');
    PERFORM assert_query_row_count(1, 'SELECT * FROM array_test WHERE float4_array_1d @> ARRAY[7.0, 8.0, 9.0]::float4[]');

    -- Test array overlap operator (&&) for float4 1D arrays
    PERFORM assert_query_row_count(1, 'SELECT * FROM array_test WHERE float4_array_1d && ARRAY[1.0, 100.0]::float4[]');
    PERFORM assert_query_row_count(2, 'SELECT * FROM array_test WHERE float4_array_1d && ARRAY[10.0, 100.0]::float4[]');
    PERFORM assert_query_row_count(3, 'SELECT * FROM array_test WHERE float4_array_1d && ARRAY[1.0, 10.0]::float4[]');

    -- Test array length functions for float4 1D arrays
    PERFORM assert_query_row_count(7, 'SELECT * FROM array_test WHERE array_length(float4_array_1d, 1) = 3');
    PERFORM assert_query_row_count(0, 'SELECT * FROM array_test WHERE array_length(float4_array_1d, 1) = 4');

    -- Test array length functions for float4 2D arrays
    -- Not implemented Error: array_length for lists with dimensions other than 1 not implemented
    EXECUTE 'SET LOCAL pg_deeplake.use_deeplake_executor = off';
    PERFORM assert_query_row_count(7, 'SELECT * FROM array_test WHERE array_length(float4_array_2d, 1) = 2');
    PERFORM assert_query_row_count(7, 'SELECT * FROM array_test WHERE array_length(float4_array_2d, 2) = 2');
    EXECUTE 'RESET pg_deeplake.use_deeplake_executor';

    -- Test array length functions for bytea 1D arrays
    PERFORM assert_query_row_count(7, 'SELECT * FROM array_test WHERE array_length(bytea_array_1d, 1) = 3');

    -- Test ANY operator with float4 1D arrays
    PERFORM assert_query_row_count(1, 'SELECT * FROM array_test WHERE 1.0 = ANY(float4_array_1d)');
    PERFORM assert_query_row_count(2, 'SELECT * FROM array_test WHERE 10.0 = ANY(float4_array_1d)');
    PERFORM assert_query_row_count(1, 'SELECT * FROM array_test WHERE 15.0 = ANY(float4_array_1d)');

    -- Test array element access for text 1D arrays
    PERFORM assert_query_row_count(1, 'SELECT * FROM array_test WHERE text_array_1d[1] = ''text1''::text');
    PERFORM assert_query_row_count(1, 'SELECT * FROM array_test WHERE text_array_1d[2] = ''text2''::text');
    PERFORM assert_query_row_count(1, 'SELECT * FROM array_test WHERE text_array_1d[3] = ''text3''::text');

    -- Test array element access for varchar 1D arrays
    PERFORM assert_query_row_count(1, 'SELECT * FROM array_test WHERE varchar_array_1d[1] = ''varchar1''::varchar');
    PERFORM assert_query_row_count(1, 'SELECT * FROM array_test WHERE varchar_array_1d[2] = ''varchar2''::varchar');
    PERFORM assert_query_row_count(1, 'SELECT * FROM array_test WHERE varchar_array_1d[3] = ''varchar3''::varchar');

    -- Bulk insert with array generation
    INSERT INTO array_test
    SELECT
        ARRAY[i::float4, (i+1)::float4, (i+2)::float4],
        ARRAY[[i::float4, (i+1)::float4], [(i+2)::float4, (i+3)::float4]],
        ARRAY[('\x' || lpad(to_hex(i % 256), 2, '0'))::bytea, ('\x' || lpad(to_hex((i+1) % 256), 2, '0'))::bytea, ('\x' || lpad(to_hex((i+2) % 256), 2, '0'))::bytea],
        ARRAY['text' || i::text, 'text' || (i+1)::text, 'text' || (i+2)::text],
        ARRAY['varchar' || i::text, 'varchar' || (i+1)::text, 'varchar' || (i+2)::text]
    FROM generate_series(1, 1000) i;

    PERFORM assert_table_row_count(1007, 'array_test');

    -- Test filtering on bulk inserted data
    PERFORM assert_query_row_count(1, 'SELECT * FROM array_test WHERE float4_array_1d[1] = 100.0');
    PERFORM assert_query_row_count(1, 'SELECT * FROM array_test WHERE float4_array_1d[2] = 501.0');
    PERFORM assert_query_row_count(5, 'SELECT * FROM array_test WHERE 10.0 = ANY(float4_array_1d)');

    RAISE NOTICE 'Test passed';
    EXCEPTION
    WHEN OTHERS THEN
        RAISE NOTICE 'ERROR: Test failed: %', SQLERRM;
END $$;

-- Cleanup
DROP TABLE IF EXISTS array_test CASCADE;
DROP EXTENSION IF EXISTS pg_deeplake CASCADE;
RESET pg_deeplake.use_deeplake_executor;