"""
Test hybrid search combining vector embeddings and text search.

Ported from: postgres/tests/sql/hybrid_search.sql
"""
import pytest
import asyncpg
from lib.assertions import Assertions


@pytest.mark.asyncio
@pytest.mark.slow
async def test_hybrid_search(db_conn: asyncpg.Connection):
    """
    Test hybrid search with cosine, maxsim, and BM25 similarity.

    Tests:
    - Cosine similarity on 1D embeddings
    - Maxsim similarity on 2D embeddings
    - BM25 text similarity on keywords
    - Hybrid search (embedding + keywords)
    - Hybrid search with custom weights
    - Index scan verification for all search types
    """
    assertions = Assertions(db_conn)

    try:
        # Create table with 1D embeddings, 2D embeddings, and keywords
        await db_conn.execute("""
            CREATE TABLE documents (
                id SERIAL PRIMARY KEY,
                title text,
                content text,
                embedding_1d float4[],
                embedding_2d float4[][],
                keywords text
            ) USING deeplake
        """)

        # Insert 25 test documents with varying embeddings
        await db_conn.execute("""
            INSERT INTO documents (title, content, embedding_1d, embedding_2d, keywords) VALUES
             ('Machine Learning Basics',
              'Introduction to machine learning algorithms and techniques',
              ARRAY[0.1, 0.2, 0.3, 0.4, 0.5],
              ARRAY[[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                    [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                    [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]],
              'machine learning algorithms introduction'),
             ('Data Science Guide',
              'Comprehensive guide to data science methodologies',
              ARRAY[0.2, 0.3, 0.4, 0.5, 0.6],
              ARRAY[[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                    [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1],
                    [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]],
              'data science guide methodologies'),
             ('AI and Neural Networks',
              'Deep dive into artificial intelligence and neural networks',
              ARRAY[0.3, 0.4, 0.5, 0.6, 0.7],
              ARRAY[[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                    [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
                    [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]],
              'artificial intelligence neural networks deep learning'),
             ('Python Programming',
              'Learn Python programming from basics to advanced',
              ARRAY[0.4, 0.5, 0.6, 0.7, 0.8],
              ARRAY[[0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1],
                    [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3],
                    [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]],
              'python programming tutorial advanced'),
             ('Database Design',
              'Principles of database design and optimization',
              ARRAY[0.5, 0.6, 0.7, 0.8, 0.9],
              ARRAY[[0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
                    [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4],
                    [0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]],
              'database design optimization principles'),
             ('Natural Language Processing',
              'Study of NLP techniques and applications in AI',
              ARRAY[0.6, 0.7, 0.8, 0.9, 1.0],
              ARRAY[[0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3],
                    [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
                    [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7]],
              'nlp natural language processing ai'),
             ('Cloud Computing',
              'Introduction to cloud infrastructure and distributed systems',
              ARRAY[0.7, 0.8, 0.9, 1.0, 1.1],
              ARRAY[[0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4],
                    [0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
                    [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8]],
              'cloud computing infrastructure distributed systems'),
             ('Computer Vision',
              'Techniques for image recognition and object detection',
              ARRAY[0.8, 0.9, 1.0, 1.1, 1.2],
              ARRAY[[0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
                    [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7],
                    [1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]],
              'computer vision image recognition object detection'),
              ('Big Data Architecture',
               'Designing scalable systems for big data processing',
               ARRAY[0.9, 1.0, 1.1, 1.2, 1.3],
               ARRAY[[0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
                     [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8],
                     [1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]],
               'big data architecture scalability'),
              ('Reinforcement Learning',
               'Exploring agent-based learning and decision-making',
               ARRAY[1.0, 1.1, 1.2, 1.3, 1.4],
               ARRAY[[1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7],
                     [1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9],
                     [1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1]],
               'reinforcement learning agents decision making')
        """)

        # Insert remaining 15 documents (truncated for brevity - following same pattern)
        await db_conn.execute("""
            INSERT INTO documents (title, content, embedding_1d, embedding_2d, keywords) VALUES
              ('DevOps Practices', 'Automation and CI/CD pipelines for modern software delivery',
               ARRAY[1.1, 1.2, 1.3, 1.4, 1.5],
               ARRAY[[1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8], [1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0], [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2]],
               'devops automation ci/cd pipelines'),
              ('Cybersecurity Fundamentals', 'Understanding threats, vulnerabilities, and protection strategies',
               ARRAY[1.2, 1.3, 1.4, 1.5, 1.6],
               ARRAY[[1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9], [1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1], [1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3]],
               'cybersecurity threats vulnerabilities protection'),
              ('Edge Computing', 'Processing data closer to the source for low-latency applications',
               ARRAY[1.3, 1.4, 1.5, 1.6, 1.7],
               ARRAY[[1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0], [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2], [1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4]],
               'edge computing low latency data processing'),
              ('Graph Databases', 'Modeling and querying relationships using graph structures',
               ARRAY[1.4, 1.5, 1.6, 1.7, 1.8],
               ARRAY[[1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1], [1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3], [1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5]],
               'graph databases relationships queries'),
              ('Quantum Computing', 'Introduction to quantum algorithms and qubits',
               ARRAY[1.5, 1.6, 1.7, 1.8, 1.9],
               ARRAY[[1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2], [1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4], [1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6]],
               'quantum computing algorithms qubits'),
              ('Explainable AI', 'Making machine learning models interpretable and transparent',
               ARRAY[1.6, 1.7, 1.8, 1.9, 2.0],
               ARRAY[[1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3], [1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5], [2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7]],
               'explainable ai model interpretability'),
              ('Time Series Analysis', 'Techniques for analyzing temporal data and forecasting',
               ARRAY[1.7, 1.8, 1.9, 2.0, 2.1],
               ARRAY[[1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4], [1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6], [2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]],
               'time series analysis forecasting temporal'),
              ('Generative Models', 'Understanding GANs and VAEs for synthetic data generation',
               ARRAY[1.8, 1.9, 2.0, 2.1, 2.2],
               ARRAY[[1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5], [2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7], [2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9]],
               'generative models gans vae synthetic data'),
              ('Feature Engineering', 'Creating meaningful features to improve model performance',
               ARRAY[1.9, 2.0, 2.1, 2.2, 2.3],
               ARRAY[[1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6], [2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8], [2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0]],
               'feature engineering model performance'),
              ('Model Deployment', 'Strategies for deploying ML models to production environments',
               ARRAY[2.0, 2.1, 2.2, 2.3, 2.4],
               ARRAY[[2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7], [2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8], [2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9]],
               'model deployment production environments'),
              ('AI Ethics', 'Ethical considerations in AI development and implementation',
               ARRAY[2.1, 2.2, 2.3, 2.4, 2.5],
               ARRAY[[2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8], [2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0], [2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2]],
               'ai ethics ethical considerations'),
              ('Data Privacy', 'Principles and techniques for protecting sensitive information',
               ARRAY[2.2, 2.3, 2.4, 2.5, 2.6],
               ARRAY[[2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9], [2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1], [2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3]],
               'data privacy principles techniques'),
              ('Robotics and Automation', 'Integration of AI in robotics and automation systems',
               ARRAY[2.3, 2.4, 2.5, 2.6, 2.7],
               ARRAY[[2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0], [2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2], [2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4]],
               'robotics automation ai integration'),
              ('AI-Driven Decision Making', 'Using AI to support complex decision-making processes',
               ARRAY[2.4, 2.5, 2.6, 2.7, 2.8],
               ARRAY[[2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1], [2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3], [2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5]],
               'ai-driven decision making complex processes'),
              ('AI in Healthcare', 'Applications of AI in healthcare and medical research',
               ARRAY[2.5, 2.6, 2.7, 2.8, 2.9],
               ARRAY[[2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2], [2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4], [2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6]],
               'ai in healthcare medical research'),
              ('AI in Finance', 'Applications of AI in finance and banking',
               ARRAY[2.6, 2.7, 2.8, 2.9, 3.0],
               ARRAY[[2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3], [2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5], [3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7]],
               'ai in finance banking financial services')
        """)

        # Create indexes for hybrid search
        await db_conn.execute("""
            CREATE INDEX idx_hybrid_docs_cosine ON documents
            USING deeplake_index(((embedding_1d, keywords)::deeplake_hybrid_record) deeplake_hybrid_ops DESC)
        """)

        await db_conn.execute("""
            CREATE INDEX index_for_cosine ON documents
            USING deeplake_index (embedding_1d DESC)
        """)

        await db_conn.execute("""
            CREATE INDEX index_for_keywords ON documents
            USING deeplake_index (keywords DESC)
        """)

        await db_conn.execute("""
            CREATE INDEX idx_hybrid_docs_maxsim ON documents
            USING deeplake_index(((embedding_2d, keywords)::deeplake_hybrid_record) deeplake_hybrid_ops DESC)
        """)

        await db_conn.execute("""
            CREATE INDEX index_for_maxsim ON documents
            USING deeplake_index (embedding_2d DESC)
        """)

        # --- Test Cosine, Maxsim and BM25 Similarity ---

        # Test 1: Cosine similarity on 1D embeddings (index scan verification)
        explain_cosine = await db_conn.fetch("""
            EXPLAIN SELECT id, title, content, embedding_1d <#> ARRAY[0.1, 0.2, 0.3, 0.4, 0.5] as score
            FROM documents ORDER BY score LIMIT 10
        """)
        explain_cosine_text = "\n".join([r[0] for r in explain_cosine])
        assert ("Index Scan" in explain_cosine_text or "Bitmap" in explain_cosine_text), \
            f"Query should use index scan (cosine)! Got: {explain_cosine_text}"

        # Test 2: Maxsim similarity on 2D embeddings (index scan verification)
        explain_maxsim = await db_conn.fetch("""
            EXPLAIN SELECT id, title, content, embedding_2d <#> ARRAY[[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]] as score
            FROM documents ORDER BY score LIMIT 10
        """)
        explain_maxsim_text = "\n".join([r[0] for r in explain_maxsim])
        assert ("Index Scan" in explain_maxsim_text or "Bitmap" in explain_maxsim_text), \
            f"Query should use index scan (maxsim)! Got: {explain_maxsim_text}"

        # Test 3: BM25 keyword search (index scan verification)
        explain_keywords = await db_conn.fetch("""
            EXPLAIN SELECT id, title, content, keywords <#> 'machine learning' as score
            FROM documents ORDER BY score LIMIT 10
        """)
        explain_keywords_text = "\n".join([r[0] for r in explain_keywords])
        assert ("Index Scan" in explain_keywords_text or "Bitmap" in explain_keywords_text), \
            f"Query should use index scan (keywords)! Got: {explain_keywords_text}"

        # Test actual cosine results with expected values
        cosine_results = await db_conn.fetch("""
            SELECT id, title, content, embedding_1d <#> ARRAY[0.1, 0.2, 0.3, 0.4, 0.5] as score
            FROM documents ORDER BY score LIMIT 10
        """)

        # Verify top result is id=1 (Machine Learning Basics)
        assert cosine_results[0]['id'] == 1, \
            f"Expected top result id=1, got {cosine_results[0]['id']}"
        assert abs(cosine_results[0]['score'] - 1.0) < 1e-3, \
            f"Expected score ~1.0, got {cosine_results[0]['score']}"

        # --- Test Hybrid Search with Cosine and BM25 ---

        # Test hybrid search with default weights
        explain_hybrid = await db_conn.fetch("""
            EXPLAIN SELECT id, title, content,
                (embedding_1d, keywords) <#> deeplake_hybrid_record(ARRAY[0.1, 0.2, 0.3, 0.4, 0.5], 'machine learning') AS score
            FROM documents ORDER BY score LIMIT 10
        """)
        explain_hybrid_text = "\n".join([r[0] for r in explain_hybrid])
        assert ("Index Scan" in explain_hybrid_text or "Bitmap" in explain_hybrid_text), \
            f"Query should use index scan (hybrid cosine)! Got: {explain_hybrid_text}"

        # Test hybrid search with custom weights (0.999, 0.001)
        explain_hybrid_weighted = await db_conn.fetch("""
            EXPLAIN SELECT id, title, content,
                (embedding_1d, keywords) <#> deeplake_hybrid_record(ARRAY[0.1, 0.2, 0.3, 0.4, 0.5], 'machine learning', 0.999, 0.001) AS score
            FROM documents ORDER BY score LIMIT 10
        """)
        explain_hybrid_weighted_text = "\n".join([r[0] for r in explain_hybrid_weighted])
        assert ("Index Scan" in explain_hybrid_weighted_text or "Bitmap" in explain_hybrid_weighted_text), \
            f"Query should use index scan (hybrid cosine weighted)! Got: {explain_hybrid_weighted_text}"

        # Test hybrid search with maxsim
        explain_hybrid_maxsim = await db_conn.fetch("""
            EXPLAIN SELECT id, title, content,
                (embedding_2d, keywords) <#> deeplake_hybrid_record(ARRAY[[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]], 'machine learning') as score
            FROM documents ORDER BY score LIMIT 10
        """)
        explain_hybrid_maxsim_text = "\n".join([r[0] for r in explain_hybrid_maxsim])
        assert ("Index Scan" in explain_hybrid_maxsim_text or "Bitmap" in explain_hybrid_maxsim_text), \
            f"Query should use index scan (hybrid maxsim)! Got: {explain_hybrid_maxsim_text}"

        # Run actual hybrid queries and verify top result
        hybrid_results = await db_conn.fetch("""
            SELECT id, title, content,
                (embedding_1d, keywords) <#> deeplake_hybrid_record(ARRAY[0.1, 0.2, 0.3, 0.4, 0.5], 'machine learning') AS score
            FROM documents ORDER BY score LIMIT 10
        """)

        # Verify top result is id=1 (Machine Learning Basics) with highest score
        assert hybrid_results[0]['id'] == 1, \
            f"Expected top hybrid result id=1, got {hybrid_results[0]['id']}"
        assert hybrid_results[0]['score'] > hybrid_results[1]['score'], \
            f"Expected top result to have highest score"

        print("✓ Test passed: Hybrid search with cosine, maxsim, and BM25 works correctly")

    finally:
        # Cleanup
        await db_conn.execute("DROP TABLE IF EXISTS documents CASCADE")
