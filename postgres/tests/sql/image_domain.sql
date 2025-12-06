\i sql/utils.psql

DROP TABLE IF EXISTS array_test CASCADE;
DROP EXTENSION IF EXISTS pg_deeplake CASCADE;
CREATE EXTENSION pg_deeplake;

-- Test IMAGE domain type (domain over BYTEA)
DO $$
BEGIN
    -- Test basic IMAGE column creation
    CREATE TABLE image_test (
        id INT,
        photo IMAGE,
        thumbnail IMAGE
    ) USING deeplake;

    -- Test basic inserts with IMAGE domain
    INSERT INTO image_test (id, photo, thumbnail)
    VALUES
        (1, '\x89504E47'::IMAGE, '\xFF'::IMAGE),  -- Simple binary data
        (2, '\xDEADBEEF'::IMAGE, '\xCAFEBABE'::IMAGE),
        (3, '\x00010203'::IMAGE, '\x04050607'::IMAGE);

    -- Test NULL handling
    INSERT INTO image_test (id, photo, thumbnail)
    VALUES
        (4, NULL, '\x08090A0B'::IMAGE),
        (5, '\x0C0D0E0F'::IMAGE, NULL),
        (6, NULL, NULL);

    -- Test bulk insert with IMAGE domain
    INSERT INTO image_test (id, photo, thumbnail)
    SELECT
        i + 10,
        ('\x' || lpad(to_hex(i), 8, '0'))::IMAGE,
        ('\x' || lpad(to_hex(i * 2), 8, '0'))::IMAGE
    FROM generate_series(1, 5) AS i;

    -- Test exact matches for IMAGE columns
    PERFORM assert_query_row_count(1, 'SELECT * FROM image_test WHERE photo = FROM_HEX(''89504E47'')');
    PERFORM assert_query_row_count(1, 'SELECT * FROM image_test WHERE thumbnail = FROM_HEX(''FF'')');

    -- Test NULL filtering
    PERFORM assert_query_row_count(2, 'SELECT * FROM image_test WHERE photo IS NULL');
    PERFORM assert_query_row_count(2, 'SELECT * FROM image_test WHERE thumbnail IS NULL');
    PERFORM assert_query_row_count(9, 'SELECT * FROM image_test WHERE photo IS NOT NULL');
    PERFORM assert_query_row_count(9, 'SELECT * FROM image_test WHERE thumbnail IS NOT NULL');

    -- Test that IMAGE behaves like BYTEA
    -- Length function should work
    PERFORM assert_query_row_count(1, 'SELECT * FROM image_test WHERE octet_length(photo) = 4 AND id = 1');
    PERFORM assert_query_row_count(1, 'SELECT * FROM image_test WHERE octet_length(photo) = 4 AND id = 2');

    -- Test comparison operators
    PERFORM assert_query_row_count(1, 'SELECT * FROM image_test WHERE photo > FROM_HEX(''00000000'') AND id = 1');

    -- Test that IMAGE and BYTEA are compatible (can compare/cast)
    EXECUTE 'SET LOCAL pg_deeplake.use_deeplake_executor = off';
    PERFORM assert_query_row_count(1, 'SELECT * FROM image_test WHERE photo = FROM_HEX(''89504E47'')::BYTEA AND id = 1');
    PERFORM assert_query_row_count(1, 'SELECT * FROM image_test WHERE photo::BYTEA = FROM_HEX(''89504E47'')::BYTEA AND id = 1');
    EXECUTE 'RESET pg_deeplake.use_deeplake_executor';

    RAISE NOTICE 'IMAGE domain tests passed';
    EXCEPTION
    WHEN OTHERS THEN
        RAISE NOTICE 'ERROR: Test failed: %', SQLERRM;
END;
$$;

-- Cleanup
DROP TABLE IF EXISTS image_test CASCADE;
DROP EXTENSION IF EXISTS pg_deeplake CASCADE;
RESET pg_deeplake.use_deeplake_executor;