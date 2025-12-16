"""
Test text search functionality with various index types.

Ported from: postgres/tests/sql/text_search.sql
"""
import pytest
import asyncpg
from test_utils.assertions import Assertions


@pytest.mark.asyncio
async def test_text_search(db_conn: asyncpg.Connection):
    """
    Test text search with inverted, exact_text, and bm25 indexes.

    Tests:
    - Creating inverted index on id column
    - Creating exact_text index on tags column
    - Creating bm25 index on content column
    - Exact match searches (= operator)
    - Contains searches (contains() function)
    - BM25 similarity search (ORDER BY <#>)
    - Index scan verification
    - Non-existent value queries
    """
    assertions = Assertions(db_conn)

    try:
        # Disable deeplake executor to use standard PostgreSQL executor
        await db_conn.execute("SET pg_deeplake.use_deeplake_executor = false")

        # Create table
        await db_conn.execute("""
            CREATE TABLE documents (
                id SERIAL PRIMARY KEY,
                title text,
                content text,
                tags text,
                category text
            ) USING deeplake
        """)

        # Insert test documents
        await db_conn.execute("""
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
                 'Data Science')
        """)

        # Create inverted index on id column
        await db_conn.execute("""
            CREATE INDEX idx_id_inverted ON documents
            USING deeplake_index (id) WITH (index_type = 'inverted')
        """)

        # Create exact_text index on tags column
        await db_conn.execute("""
            CREATE INDEX idx_tags_exact ON documents
            USING deeplake_index (tags) WITH (index_type = 'exact_text')
        """)

        # Create bm25 index on content column
        await db_conn.execute("""
            CREATE INDEX idx_content_bm25 ON documents
            USING deeplake_index (content) WITH (index_type = 'bm25')
        """)

        # Test 1: Exact match using = operator on tags column
        explain_test1 = await db_conn.fetch("""
            EXPLAIN SELECT id, title, tags FROM documents
            WHERE tags = 'machine learning algorithms introduction'
            LIMIT 10
        """)
        explain_text1 = "\n".join([row[0] for row in explain_test1])

        # Note: May use Index Scan or Bitmap Index Scan
        has_index_scan1 = ("Index Scan" in explain_text1 or "Bitmap" in explain_text1)
        assert has_index_scan1, \
            f"Query should use index scan for exact text search. Got: {explain_text1}"

        result_test1 = await db_conn.fetch("""
            SELECT id, title, tags FROM documents
            WHERE tags = 'machine learning algorithms introduction'
            LIMIT 10
        """)
        assert len(result_test1) == 1, \
            f"Expected 1 result for exact match, got {len(result_test1)}"

        # Test 2: Different exact match on tags
        explain_test2 = await db_conn.fetch("""
            EXPLAIN SELECT id, title, tags FROM documents
            WHERE id <= 4 AND tags = 'python programming tutorial advanced'
            LIMIT 10
        """)
        explain_text2 = "\n".join([row[0] for row in explain_test2])

        has_index_scan2 = ("Index Scan" in explain_text2 or "Bitmap" in explain_text2)
        assert has_index_scan2, \
            f"Query should use index scan for exact text search. Got: {explain_text2}"

        result_test2 = await db_conn.fetch("""
            SELECT id, title, tags FROM documents
            WHERE id <= 4 AND tags = 'python programming tutorial advanced'
            LIMIT 10
        """)
        assert len(result_test2) == 1, \
            f"Expected 1 result for exact match, got {len(result_test2)}"

        # Test 3: Contains search using contains() function
        explain_test3 = await db_conn.fetch("""
            EXPLAIN SELECT id, title, tags FROM documents
            WHERE contains(tags, 'machine')
            LIMIT 10
        """)
        explain_text3 = "\n".join([row[0] for row in explain_test3])

        has_index_scan3 = ("Index Scan" in explain_text3 or "Bitmap" in explain_text3)
        assert has_index_scan3, \
            f"Query should use index scan for contains search. Got: {explain_text3}"

        # Note: Test 3 result verification is disabled in original SQL test
        # due to known issue with deeplake index handling contains queries
        result_test3 = await db_conn.fetch("""
            SELECT id, title, tags FROM documents
            WHERE contains(tags, 'machine')
            LIMIT 10
        """)
        # Expected 1 result but not verified due to known issue

        # Test 4: BM25 similarity search on content column (ORDER BY)
        explain_test4 = await db_conn.fetch("""
            EXPLAIN SELECT id, title, content FROM documents
            ORDER BY content <#> 'machine learning algorithms'
            LIMIT 3
        """)
        explain_text4 = "\n".join([row[0] for row in explain_test4])

        has_index_scan4 = ("Index Scan" in explain_text4 or "Bitmap" in explain_text4)
        assert has_index_scan4, \
            f"Query should use index scan for BM25 search. Got: {explain_text4}"

        result_test4 = await db_conn.fetch("""
            SELECT id, title, content FROM documents
            ORDER BY content <#> 'machine learning algorithms'
            LIMIT 3
        """)
        assert len(result_test4) == 1, \
            f"Expected 1 result for BM25 search, got {len(result_test4)}"

        # Test 5: Non-existent exact match returns empty results
        result_test5 = await db_conn.fetch("""
            SELECT id, title, tags FROM documents
            WHERE tags = 'nonexistent tag value'
            LIMIT 10
        """)
        assert len(result_test5) == 0, \
            f"Expected 0 results for non-existent value, got {len(result_test5)}"

        # Test 6: Query without index (category column)
        result_test6 = await db_conn.fetch("""
            SELECT id, title, category FROM documents
            WHERE category = 'AI'
            LIMIT 10
        """)
        assert len(result_test6) == 2, \
            f"Expected 2 results for category='AI', got {len(result_test6)}"

        print("✓ Test passed: Text search with inverted, exact_text, and bm25 indexes works correctly")

    finally:
        # Cleanup
        await db_conn.execute("DROP TABLE IF EXISTS documents CASCADE")
        await db_conn.execute("RESET pg_deeplake.use_deeplake_executor")
