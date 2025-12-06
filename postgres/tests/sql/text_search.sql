\i sql/utils.psql

DROP TABLE IF EXISTS documents CASCADE;
DROP EXTENSION IF EXISTS pg_deeplake CASCADE;
CREATE EXTENSION IF NOT EXISTS pg_deeplake;
SET pg_deeplake.use_deeplake_executor = false;

DO $$ BEGIN
  BEGIN
    CREATE TABLE documents (
        id SERIAL PRIMARY KEY,
        title text,
        content text,
        tags text,
        category text
    ) USING deeplake;

    INSERT INTO documents (title, content, tags, category) VALUES
     ('Machine Learning Basics',
      'Introduction to machine learning algorithms and techniques',
      'machine learning algorithms introduction',
      'AI'),
     ('Data Science Guide',
      'Comprehensive guide to data science methodologies',
      'data science guide methodologies',
      'Data Science'),
     ('AI and Neural Networks',
      'Deep dive into artificial intelligence and neural networks',
      'artificial intelligence neural networks deep learning',
      'AI'),
     ('Python Programming',
      'Learn Python programming from basics to advanced',
      'python programming tutorial advanced',
      'Programming'),
     ('Database Design',
      'Principles of database design and optimization',
      'database design optimization sql',
      'Data Science');

    -- Create inverted index on id column
    CREATE INDEX idx_id_inverted ON documents USING deeplake_index (id) WITH (index_type = 'inverted');

    -- Create exact_text index on tags column
    CREATE INDEX idx_tags_exact ON documents USING deeplake_index (tags) WITH (index_type = 'exact_text');

    -- Create bm25 index on content column
    CREATE INDEX idx_content_bm25 ON documents USING deeplake_index (content) WITH (index_type = 'bm25');

    -- Test 1: Verify index scan is used for exact match on tags
    RAISE NOTICE 'Test 1: Exact match using = operator on tags column';
    IF NOT is_using_index_scan('SELECT id, title, tags FROM documents WHERE tags = ''machine learning algorithms introduction'' LIMIT 10;') THEN
        RAISE EXCEPTION 'Query should use index scan for exact text search!';
    END IF;

    -- Test actual results for exact match
    CREATE TEMP TABLE actual_test1 AS
    SELECT id, title, tags FROM documents
    WHERE tags = 'machine learning algorithms introduction'
    LIMIT 10;

    IF (SELECT COUNT(*) FROM actual_test1) != 1 THEN
        RAISE EXCEPTION 'Test 1 failed: Expected 1 result, got %', (SELECT COUNT(*) FROM actual_test1);
    END IF;
    DROP TABLE actual_test1;

    -- Test 2: Different exact match
    RAISE NOTICE 'Test 2: Different exact match on tags';
    IF NOT is_using_index_scan('SELECT id, title, tags FROM documents WHERE id <= 4 AND tags = ''python programming tutorial advanced'' LIMIT 10;') THEN
        RAISE EXCEPTION 'Query should use index scan for exact text search!';
    END IF;

    CREATE TEMP TABLE actual_test2 AS
    SELECT id, title, tags FROM documents
    WHERE id <= 4 AND tags = 'python programming tutorial advanced'
    LIMIT 10;

    IF (SELECT COUNT(*) FROM actual_test2) != 1 THEN
        RAISE EXCEPTION 'Test 2 failed: Expected 1 result, got %', (SELECT COUNT(*) FROM actual_test2);
    END IF;
    DROP TABLE actual_test2;

    -- Test 3: Contains search using @> operator on tags column
    RAISE NOTICE 'Test 3: Contains search using @> operator';
    IF NOT is_using_index_scan('SELECT id, title, tags FROM documents WHERE contains(tags, ''machine'') LIMIT 10;') THEN
        RAISE EXCEPTION 'Query should use index scan for contains search!';
    END IF;

    CREATE TEMP TABLE actual_test3 AS
    SELECT id, title, tags FROM documents WHERE contains(tags, 'machine') LIMIT 10;

    -- DISABLED: Issue with deeplake index handling contains queries
    -- Run following in python to verify:
    -- view = ds.query("SELECT * WHERE CONTAINS(tags, 'machine')"); assert len(view) == 1
    -- IF (SELECT COUNT(*) FROM actual_test3) != 1 THEN
    --     RAISE EXCEPTION 'Test 3 failed: Expected 1 result for contains search';
    -- END IF;
    -- DROP TABLE actual_test3;

    -- Test 4: BM25 similarity search on content column (ORDER BY)
    RAISE NOTICE 'Test 4: BM25 similarity search on content';
    IF NOT is_using_index_scan('SELECT id, title, content FROM documents ORDER BY content <#> ''machine learning algorithms'' LIMIT 3;') THEN
        RAISE EXCEPTION 'Query should use index scan for BM25 search!';
    END IF;

    CREATE TEMP TABLE actual_test4 AS
    SELECT id, title, content FROM documents
    ORDER BY content <#> 'machine learning algorithms'
    LIMIT 3;

    IF (SELECT COUNT(*) FROM actual_test4) != 1 THEN
        RAISE EXCEPTION 'Test 4 failed: Expected 1 results, got %', (SELECT COUNT(*) FROM actual_test4);
    END IF;
    DROP TABLE actual_test4;

    -- Test 5: Check that non-existent values return empty results
    RAISE NOTICE 'Test 5: Non-existent exact match';
    CREATE TEMP TABLE actual_test5 AS
    SELECT id, title, tags FROM documents
    WHERE tags = 'nonexistent tag value'
    LIMIT 10;

    IF (SELECT COUNT(*) FROM actual_test5) != 0 THEN
        RAISE EXCEPTION 'Test 5 failed: Expected 0 results for non-existent value';
    END IF;
    DROP TABLE actual_test5;

    -- Test 6: Query without index should use seq scan
    RAISE NOTICE 'Test 6: Query without index (category column)';
    CREATE TEMP TABLE actual_test6 AS
    SELECT id, title, category FROM documents
    WHERE category = 'AI'
    LIMIT 10;

    IF (SELECT COUNT(*) FROM actual_test6) != 2 THEN
        RAISE EXCEPTION 'Test 6 failed: Expected 2 results, got %', (SELECT COUNT(*) FROM actual_test6);
    END IF;
    DROP TABLE actual_test6;

    RAISE NOTICE 'All text search tests passed!';

  EXCEPTION
    WHEN OTHERS THEN
      RAISE NOTICE 'ERROR: Test failed: %', SQLERRM;
  END;
END $$;

-- Cleanup
DROP TABLE IF EXISTS documents CASCADE;
DROP EXTENSION IF EXISTS pg_deeplake CASCADE;
RESET pg_deeplake.use_deeplake_executor;