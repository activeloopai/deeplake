\i sql/utils.psql

DROP TABLE IF EXISTS documents CASCADE;
DROP EXTENSION IF EXISTS pg_deeplake CASCADE;
CREATE EXTENSION IF NOT EXISTS pg_deeplake;

DO $$ BEGIN
  BEGIN
    CREATE TABLE documents (
        id SERIAL PRIMARY KEY,
        title text,
        content text,
        embedding_1d float4[],
        embedding_2d float4[][],
        keywords text
    ) USING deeplake;

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
       'reinforcement learning agents decision making'),

      ('DevOps Practices',
       'Automation and CI/CD pipelines for modern software delivery',
       ARRAY[1.1, 1.2, 1.3, 1.4, 1.5],
       ARRAY[[1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8],
             [1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0],
             [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2]],
       'devops automation ci/cd pipelines'),

      ('Cybersecurity Fundamentals',
       'Understanding threats, vulnerabilities, and protection strategies',
       ARRAY[1.2, 1.3, 1.4, 1.5, 1.6],
       ARRAY[[1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9],
             [1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1],
             [1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3]],
       'cybersecurity threats vulnerabilities protection'),

      ('Edge Computing',
       'Processing data closer to the source for low-latency applications',
       ARRAY[1.3, 1.4, 1.5, 1.6, 1.7],
       ARRAY[[1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0],
             [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2],
             [1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4]],
       'edge computing low latency data processing'),

      ('Graph Databases',
       'Modeling and querying relationships using graph structures',
       ARRAY[1.4, 1.5, 1.6, 1.7, 1.8],
       ARRAY[[1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1],
             [1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3],
             [1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5]],
       'graph databases relationships queries'),

      ('Quantum Computing',
       'Introduction to quantum algorithms and qubits',
       ARRAY[1.5, 1.6, 1.7, 1.8, 1.9],
       ARRAY[[1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2],
             [1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4],
             [1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6]],
       'quantum computing algorithms qubits'),

      ('Explainable AI',
       'Making machine learning models interpretable and transparent',
       ARRAY[1.6, 1.7, 1.8, 1.9, 2.0],
       ARRAY[[1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3],
             [1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5],
             [2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7]],
       'explainable ai model interpretability'),

      ('Time Series Analysis',
       'Techniques for analyzing temporal data and forecasting',
       ARRAY[1.7, 1.8, 1.9, 2.0, 2.1],
       ARRAY[[1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4],
             [1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6],
             [2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]],
       'time series analysis forecasting temporal'),

      ('Generative Models',
       'Understanding GANs and VAEs for synthetic data generation',
       ARRAY[1.8, 1.9, 2.0, 2.1, 2.2],
       ARRAY[[1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5],
             [2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7],
             [2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9]],
       'generative models gans vae synthetic data'),

      ('Feature Engineering',
       'Creating meaningful features to improve model performance',
       ARRAY[1.9, 2.0, 2.1, 2.2, 2.3],
       ARRAY[[1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6],
             [2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8],
             [2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0]],
       'feature engineering model performance'),

      ('Model Deployment',
       'Strategies for deploying ML models to production environments',
       ARRAY[2.0, 2.1, 2.2, 2.3, 2.4],
       ARRAY[[2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7],
             [2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8],
             [2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9]],
       'model deployment production environments'),

      ('AI Ethics',
       'Ethical considerations in AI development and implementation',
       ARRAY[2.1, 2.2, 2.3, 2.4, 2.5],
       ARRAY[[2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8],
             [2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0],
             [2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2]],
       'ai ethics ethical considerations'),

      ('Data Privacy',
       'Principles and techniques for protecting sensitive information',
       ARRAY[2.2, 2.3, 2.4, 2.5, 2.6],
       ARRAY[[2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9],
             [2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1],
             [2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3]],
       'data privacy principles techniques'),

      ('Robotics and Automation',
       'Integration of AI in robotics and automation systems',
       ARRAY[2.3, 2.4, 2.5, 2.6, 2.7],
       ARRAY[[2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0],
             [2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2],
             [2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4]],
       'robotics automation ai integration'),

      ('AI-Driven Decision Making',
       'Using AI to support complex decision-making processes',
       ARRAY[2.4, 2.5, 2.6, 2.7, 2.8],
       ARRAY[[2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1],
             [2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3],
             [2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5]],
       'ai-driven decision making complex processes'),

      ('AI in Healthcare',
       'Applications of AI in healthcare and medical research',
       ARRAY[2.5, 2.6, 2.7, 2.8, 2.9],
       ARRAY[[2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2],
             [2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4],
             [2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6]],
       'ai in healthcare medical research'),

      ('AI in Finance',
       'Applications of AI in finance and banking',
       ARRAY[2.6, 2.7, 2.8, 2.9, 3.0],
       ARRAY[[2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3],
             [2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5],
             [3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7]],
       'ai in finance banking financial services');

    CREATE INDEX idx_hybrid_docs_cosine ON documents USING deeplake_index(((embedding_1d, keywords)::deeplake_hybrid_record) deeplake_hybrid_ops DESC);
    CREATE INDEX index_for_cosine ON documents USING deeplake_index (embedding_1d DESC);
    CREATE INDEX index_for_keywords ON documents USING deeplake_index (keywords DESC);
    CREATE INDEX idx_hybrid_docs_maxsim ON documents USING deeplake_index(((embedding_2d, keywords)::deeplake_hybrid_record) deeplake_hybrid_ops DESC);
    CREATE INDEX index_for_maxsim ON documents USING deeplake_index (embedding_2d DESC);

----------------------------------- Cosine, Maxsim and BM25 Similarity ---------------------------------------
    -- Make sure index scan is used for cosine, maxsim and bm25 similarity.
    -- Following queries are run for testing.
    --SELECT id, title, content, embedding_1d <#> ARRAY[0.1, 0.2, 0.3, 0.4, 0.5] as score FROM documents ORDER BY score LIMIT 10;
    --SELECT id, title, content, embedding_2d <#> ARRAY[[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]] as score FROM documents ORDER BY score LIMIT 10;
    --SELECT id, title, content, keywords <#> 'machine learning' as score FROM documents ORDER BY score LIMIT 10;
    --EXPLAIN SELECT id, title, content, embedding_1d <#> ARRAY[0.1, 0.2, 0.3, 0.4, 0.5] as score FROM documents ORDER BY score LIMIT 10;
    --EXPLAIN SELECT id, title, content, embedding_2d <#> ARRAY[[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]] as score FROM documents ORDER BY score LIMIT 10;
    --EXPLAIN SELECT id, title, content, keywords <#> 'machine learning' as score FROM documents ORDER BY score LIMIT 10;
    IF NOT is_using_index_scan('SELECT id, title, content, embedding_1d <#> ARRAY[0.1, 0.2, 0.3, 0.4, 0.5] as score FROM documents ORDER BY score LIMIT 10;') THEN
        RAISE EXCEPTION 'Query should use an index scan (cosine)!';
    END IF;

    IF NOT is_using_index_scan('SELECT id, title, content, embedding_2d <#> ARRAY[[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]] as score FROM documents ORDER BY score LIMIT 10;') THEN
        RAISE EXCEPTION 'Query should use an index scan (maxsim)!';
    END IF;

    IF NOT is_using_index_scan('SELECT id, title, content, keywords <#> ''machine learning'' as score FROM documents ORDER BY score LIMIT 10;') THEN
        RAISE EXCEPTION 'Query should use an index scan (keywords)!';
    END IF;

    CREATE TEMP TABLE expected_documents (id INTEGER, title TEXT, content TEXT, score REAL);
    INSERT INTO expected_documents VALUES (1, 'Machine Learning Basics', 'Introduction to machine learning algorithms and techniques', 1),
    (2, 'Data Science Guide', 'Comprehensive guide to data science methodologies', 0.99493665),
    (3, 'AI and Neural Networks', 'Deep dive into artificial intelligence and neural networks', 0.98644006),
    (4, 'Python Programming', 'Learn Python programming from basics to advanced', 0.9782321),
    (5, 'Database Design', 'Principles of database design and optimization', 0.9710608),
    (6, 'Natural Language Processing', 'Study of NLP techniques and applications in AI', 0.96495056),
    (7, 'Cloud Computing', 'Introduction to cloud infrastructure and distributed systems', 0.95975995),
    (8, 'Computer Vision', 'Techniques for image recognition and object detection', 0.9553303),
    (9, 'Big Data Architecture', 'Designing scalable systems for big data processing', 0.95152277),
    (10, 'Reinforcement Learning', 'Exploring agent-based learning and decision-making', 0.9482238);
    CREATE TEMP TABLE actual AS SELECT id, title, content, embedding_1d <#> ARRAY[0.1, 0.2, 0.3, 0.4, 0.5] as score FROM documents ORDER BY score LIMIT 10;

    IF EXISTS (
        SELECT 1
        FROM expected_documents e
        JOIN actual a USING (id, title, content)
        WHERE abs(e.score - a.score) > 1e-6
    ) THEN
        RAISE EXCEPTION 'Test failed: floating point mismatch!';
    END IF;

    DROP TABLE actual;
    DROP TABLE expected_documents;

    CREATE TEMP TABLE expected_documents (id INTEGER, title TEXT, content TEXT, score REAL);
    INSERT INTO expected_documents VALUES (1, 'Machine Learning Basics', 'Introduction to machine learning algorithms and techniques', 0.99999994),
    (2, 'Data Science Guide', 'Comprehensive guide to data science methodologies', 0.997096),
    (3, 'AI and Neural Networks', 'Deep dive into artificial intelligence and neural networks', 0.9912933),
    (4, 'Python Programming', 'Learn Python programming from basics to advanced', 0.9848203),
    (5, 'Database Design', 'Principles of database design and optimization', 0.97851676),
    (6, 'Natural Language Processing', 'Study of NLP techniques and applications in AI', 0.97267884),
    (7, 'Cloud Computing', 'Introduction to cloud infrastructure and distributed systems', 0.96738243),
    (8, 'Computer Vision', 'Techniques for image recognition and object detection', 0.9626168),
    (9, 'Big Data Architecture', 'Designing scalable systems for big data processing', 0.9583381),
    (10, 'Reinforcement Learning', 'Exploring agent-based learning and decision-making', 0.954494);
    CREATE TEMP TABLE actual AS SELECT id, title, content, embedding_2d <#> ARRAY[[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]] as score FROM documents ORDER BY score LIMIT 10;

    IF EXISTS (
        SELECT 1
        FROM expected_documents e
        JOIN actual a USING (id, title, content)
        WHERE abs(e.score - a.score) > 1e-6
    ) THEN
        RAISE EXCEPTION 'Test failed: query result differs from expected (maxsim)!';
    END IF;

    DROP TABLE actual;
    DROP TABLE expected_documents;

    CREATE TEMP TABLE expected_documents (id INTEGER, title TEXT, content TEXT, score REAL);
    INSERT INTO expected_documents VALUES (1, 'Machine Learning Basics', 'Introduction to machine learning algorithms and techniques', 5.057182),
    (10, 'Reinforcement Learning', 'Exploring agent-based learning and decision-making', 1.8425648),
    (3, 'AI and Neural Networks', 'Deep dive into artificial intelligence and neural networks', 1.6849195);
    CREATE TEMP TABLE actual AS SELECT id, title, content, keywords <#> 'machine learning' as score FROM documents ORDER BY score LIMIT 10;

    IF EXISTS (
        SELECT 1
        FROM expected_documents e
        JOIN actual a USING (id, title, content)
        WHERE abs(e.score - a.score) > 1e-6
    ) THEN
        RAISE EXCEPTION 'Test failed: query result differs from expected (keywords)!';
    END IF;

    DROP TABLE actual;
    DROP TABLE expected_documents;
----------------------------------- End of Cosine, Maxsim and BM25 Similarity --------------------------------

----------------------------------- Hybrid Search with Cosine and BM25 Similarity --------------------------------
    -- Make sure index scan is used for hybrid search with cosine
    --SELECT id, title, content, (embedding_1d, keywords) <#> deeplake_hybrid_record(ARRAY[0.1, 0.2, 0.3, 0.4, 0.5], 'machine learning') AS score FROM documents ORDER BY score LIMIT 10;
    --SELECT id, title, content, (embedding_1d, keywords) <#> deeplake_hybrid_record(ARRAY[0.1, 0.2, 0.3, 0.4, 0.5], 'machine learning', 0.999, 0.001) AS score FROM documents ORDER BY score LIMIT 10;
    --SELECT id, title, content, (embedding_1d, keywords) <#> (ARRAY[0.1, 0.2, 0.3, 0.4, 0.5], 'machine learning', 0.4, 0.6)::deeplake_hybrid_record_weighted as score FROM documents ORDER BY score LIMIT 10;
    --EXPLAIN SELECT id, title, content, (embedding_1d, keywords) <#> deeplake_hybrid_record(ARRAY[0.1, 0.2, 0.3, 0.4, 0.5], 'machine learning') AS score FROM documents ORDER BY score LIMIT 10;
    --EXPLAIN SELECT id, title, content, (embedding_1d, keywords) <#> deeplake_hybrid_record(ARRAY[0.1, 0.2, 0.3, 0.4, 0.5], 'machine learning', 0.3, 0.7) AS score FROM documents ORDER BY score LIMIT 10;
    --EXPLAIN SELECT id, title, content, (embedding_1d, keywords) <#> (ARRAY[0.1, 0.2, 0.3, 0.4, 0.5], 'machine learning', 0.4, 0.6)::deeplake_hybrid_record_weighted as score FROM documents ORDER BY score LIMIT 10;
    IF NOT is_using_index_scan('SELECT id, title, content, (embedding_1d, keywords) <#> deeplake_hybrid_record(ARRAY[0.1, 0.2, 0.3, 0.4, 0.5], ''machine learning'') AS score FROM documents ORDER BY score LIMIT 10;') THEN
        RAISE EXCEPTION 'Query should use an index scan (hybrid cosine)!';
    END IF;

    IF NOT is_using_index_scan('SELECT id, title, content, (embedding_1d, keywords) <#> deeplake_hybrid_record(ARRAY[0.1, 0.2, 0.3, 0.4, 0.5], ''machine learning'', 0.999, 0.001) AS score FROM documents ORDER BY score LIMIT 10;') THEN
        RAISE EXCEPTION 'Query should use an index scan (hybrid cosine weighted)!';
    END IF;

    IF NOT is_using_index_scan('SELECT id, title, content, (embedding_1d, keywords) <#> (ARRAY[0.1, 0.2, 0.3, 0.4, 0.5], ''machine learning'', 0.4, 0.6)::deeplake_hybrid_record_weighted as score FROM documents ORDER BY score LIMIT 10;') THEN
        RAISE EXCEPTION 'Query should use an index scan (hybrid cosine weighted 1)!';
    END IF;

    CREATE TEMP TABLE expected_documents (id INTEGER, title TEXT, content TEXT, score REAL);
    INSERT INTO expected_documents VALUES (1, 'Machine Learning Basics', 'Introduction to machine learning algorithms and techniques', 0.51680136),
    (10, 'Reinforcement Learning', 'Exploring agent-based learning and decision-making', 0.0675575),
    (3, 'AI and Neural Networks', 'Deep dive into artificial intelligence and neural networks', 0.06673473),
    (2, 'Data Science Guide', 'Comprehensive guide to data science methodologies', 0.051201202),
    (4, 'Python Programming', 'Learn Python programming from basics to advanced', 0.050353017),
    (5, 'Database Design', 'Principles of database design and optimization', 0.04999321),
    (6, 'Natural Language Processing', 'Study of NLP techniques and applications in AI', 0.04968867),
    (7, 'Cloud Computing', 'Introduction to cloud infrastructure and distributed systems', 0.049431425),
    (8, 'Computer Vision', 'Techniques for image recognition and object detection', 0.049212944),
    (9, 'Big Data Architecture', 'Designing scalable systems for big data processing', 0.049025923);
    CREATE TEMP TABLE actual AS SELECT id, title, content, (embedding_1d, keywords) <#> deeplake_hybrid_record(ARRAY[0.1, 0.2, 0.3, 0.4, 0.5], 'machine learning') AS score FROM documents ORDER BY score LIMIT 10;
    IF EXISTS (
        SELECT 1
        FROM expected_documents e
        JOIN actual a USING (id, title, content)
        WHERE abs(e.score - a.score) > 1e-6
    ) THEN
        RAISE EXCEPTION 'Test failed: query result differs from expected (hybrid cosine)!';
    END IF;
    DROP TABLE actual;
    DROP TABLE expected_documents;

    CREATE TEMP TABLE expected_documents (id INTEGER, title TEXT, content TEXT, score REAL);
    INSERT INTO expected_documents VALUES (1, 'Machine Learning Basics', 'Introduction to machine learning algorithms and techniques', 0.103749976),
    (2, 'Data Science Guide', 'Comprehensive guide to data science methodologies', 0.1023),
    (3, 'AI and Neural Networks', 'Deep dive into artificial intelligence and neural networks', 0.10146642),
    (4, 'Python Programming', 'Learn Python programming from basics to advanced', 0.100605324),
    (5, 'Database Design', 'Principles of database design and optimization', 0.09988643),
    (6, 'Natural Language Processing', 'Study of NLP techniques and applications in AI', 0.099277966),
    (7, 'Cloud Computing', 'Introduction to cloud infrastructure and distributed systems', 0.09876399),
    (8, 'Computer Vision', 'Techniques for image recognition and object detection', 0.098327465),
    (9, 'Big Data Architecture', 'Designing scalable systems for big data processing', 0.0979538),
    (10, 'Reinforcement Learning', 'Exploring agent-based learning and decision-making', 0.09766856);
    CREATE TEMP TABLE actual AS SELECT id, title, content, (embedding_1d, keywords) <#> deeplake_hybrid_record(ARRAY[0.1, 0.2, 0.3, 0.4, 0.5], 'machine learning', 0.999, 0.001) AS score FROM documents ORDER BY score LIMIT 10;
    IF EXISTS (
        SELECT 1
        FROM expected_documents e
        JOIN actual a USING (id, title, content)
        WHERE abs(e.score - a.score) > 1e-6
    ) THEN
        RAISE EXCEPTION 'Test failed: query result differs from expected (hybrid cosine weighted)!';
    END IF;
    DROP TABLE actual;
    DROP TABLE expected_documents;

    CREATE TEMP TABLE expected_documents (id INTEGER, title TEXT, content TEXT, score REAL);
    INSERT INTO expected_documents VALUES (1, 'Machine Learning Basics', 'Introduction to machine learning algorithms and techniques', 0.5995771),
    (10, 'Reinforcement Learning', 'Exploring agent-based learning and decision-making', 0.06152322),
    (3, 'AI and Neural Networks', 'Deep dive into artificial intelligence and neural networks', 0.05977447),
    (2, 'Data Science Guide', 'Comprehensive guide to data science methodologies', 0.04096096),
    (4, 'Python Programming', 'Learn Python programming from basics to advanced', 0.040282413),
    (5, 'Database Design', 'Principles of database design and optimization', 0.039994568),
    (6, 'Natural Language Processing', 'Study of NLP techniques and applications in AI', 0.039750937),
    (7, 'Cloud Computing', 'Introduction to cloud infrastructure and distributed systems', 0.03954514),
    (8, 'Computer Vision', 'Techniques for image recognition and object detection', 0.039370354),
    (9, 'Big Data Architecture', 'Designing scalable systems for big data processing', 0.03922074);
    CREATE TEMP TABLE actual AS SELECT id, title, content, (embedding_1d, keywords) <#> (ARRAY[0.1, 0.2, 0.3, 0.4, 0.5], 'machine learning', 0.4, 0.6)::deeplake_hybrid_record_weighted as score FROM documents ORDER BY score LIMIT 10;
    IF EXISTS (
        SELECT 1
        FROM expected_documents e
        JOIN actual a USING (id, title, content)
        WHERE abs(e.score - a.score) > 1e-6
    ) THEN
        RAISE EXCEPTION 'Test failed: query result differs from expected (hybrid cosine weighted 1)!';
    END IF;
    DROP TABLE actual;
    DROP TABLE expected_documents;
----------------------------------- End of Hybrid Search with Cosine and BM25 Similarity --------------------------------


----------------------------------- Hybrid Search with Maxsim and BM25 Similarity --------------------------------
    -- Make sure index scan is used for hybrid search with maxsim
    --SELECT id, title, content, (embedding_2d, keywords) <#> deeplake_hybrid_record(ARRAY[[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]], 'machine learning') as score FROM documents ORDER BY score LIMIT 10;
    --SELECT id, title, content, (embedding_2d, keywords) <#> deeplake_hybrid_record(ARRAY[[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]], 'machine learning', 0.4, 0.6) as score FROM documents ORDER BY score LIMIT 10;
    --SELECT id, title, content, (embedding_2d, keywords) <#> (ARRAY[[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]], 'machine learning', 0.4, 0.6)::deeplake_hybrid_record_weighted as score FROM documents ORDER BY score LIMIT 10;
    --EXPLAIN SELECT id, title, content, (embedding_2d, keywords) <#> deeplake_hybrid_record(ARRAY[[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]], 'machine learning') as score FROM documents ORDER BY score LIMIT 10;
    --EXPLAIN SELECT id, title, content, (embedding_2d, keywords) <#> deeplake_hybrid_record(ARRAY[[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]], 'machine learning', 0.4, 0.6) as score FROM documents ORDER BY score LIMIT 10;
    --EXPLAIN SELECT id, title, content, (embedding_2d, keywords) <#> (ARRAY[[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]], 'machine learning', 0.4, 0.6)::deeplake_hybrid_record_weighted as score FROM documents ORDER BY score LIMIT 10;
    IF NOT is_using_index_scan('SELECT id, title, content, (embedding_2d, keywords) <#> deeplake_hybrid_record(ARRAY[[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]], ''machine learning'') as score FROM documents ORDER BY score LIMIT 10;') THEN
        RAISE EXCEPTION 'Query should use an index scan (hybrid maxsim)!';
    END IF;

    IF NOT is_using_index_scan('SELECT id, title, content, (embedding_2d, keywords) <#> deeplake_hybrid_record(ARRAY[[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]], ''machine learning'', 0.4, 0.6) as score FROM documents ORDER BY score LIMIT 10;') THEN
        RAISE EXCEPTION 'Query should use an index scan (hybrid maxsim weighted)!';
    END IF;

    IF NOT is_using_index_scan('SELECT id, title, content, (embedding_2d, keywords) <#> (ARRAY[[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]], ''machine learning'', 0.4, 0.6)::deeplake_hybrid_record_weighted as score FROM documents ORDER BY score LIMIT 10;') THEN
        RAISE EXCEPTION 'Query should use an index scan (hybrid maxsim weighted 1)!';
    END IF;

    CREATE TEMP TABLE expected_documents (id INTEGER, title TEXT, content TEXT, score REAL);
    INSERT INTO expected_documents VALUES (1, 'Machine Learning Basics', 'Introduction to machine learning algorithms and techniques', 0.5165116),
    (10, 'Reinforcement Learning', 'Exploring agent-based learning and decision-making', 0.067588024),
    (3, 'AI and Neural Networks', 'Deep dive into artificial intelligence and neural networks', 0.06669451),
    (2, 'Data Science Guide', 'Comprehensive guide to data science methodologies', 0.051023006),
    (4, 'Python Programming', 'Learn Python programming from basics to advanced', 0.05040049),
    (5, 'Database Design', 'Principles of database design and optimization', 0.05008379),
    (6, 'Natural Language Processing', 'Study of NLP techniques and applications in AI', 0.049792252),
    (7, 'Cloud Computing', 'Introduction to cloud infrastructure and distributed systems', 0.049529232),
    (8, 'Computer Vision', 'Techniques for image recognition and object detection', 0.049293756),
    (9, 'Big Data Architecture', 'Designing scalable systems for big data processing', 0.049083292);
    CREATE TEMP TABLE actual AS SELECT id, title, content, (embedding_2d, keywords) <#> deeplake_hybrid_record(ARRAY[[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]], 'machine learning') as score FROM documents ORDER BY score LIMIT 10;
    IF EXISTS (
        SELECT 1
        FROM expected_documents e
        JOIN actual a USING (id, title, content)
        WHERE abs(e.score - a.score) > 1e-6
    ) THEN
        RAISE EXCEPTION 'Test failed: query result differs from expected (hybrid maxsim)!';
    END IF;
    DROP TABLE actual;
    DROP TABLE expected_documents;

    CREATE TEMP TABLE expected_documents (id INTEGER, title TEXT, content TEXT, score REAL);
    INSERT INTO expected_documents VALUES (1, 'Machine Learning Basics', 'Introduction to machine learning algorithms and techniques', 0.5993454),
    (10, 'Reinforcement Learning', 'Exploring agent-based learning and decision-making', 0.06154764),
    (3, 'AI and Neural Networks', 'Deep dive into artificial intelligence and neural networks', 0.059742294),
    (2, 'Data Science Guide', 'Comprehensive guide to data science methodologies', 0.040818404),
    (4, 'Python Programming', 'Learn Python programming from basics to advanced', 0.040320393),
    (5, 'Database Design', 'Principles of database design and optimization', 0.040067032),
    (6, 'Natural Language Processing', 'Study of NLP techniques and applications in AI', 0.039833803),
    (7, 'Cloud Computing', 'Introduction to cloud infrastructure and distributed systems', 0.039623387),
    (8, 'Computer Vision', 'Techniques for image recognition and object detection', 0.039435007),
    (9, 'Big Data Architecture', 'Designing scalable systems for big data processing', 0.039266635);
    CREATE TEMP TABLE actual AS SELECT id, title, content, (embedding_2d, keywords) <#> deeplake_hybrid_record(ARRAY[[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]], 'machine learning', 0.4, 0.6) as score FROM documents ORDER BY score LIMIT 10;
    IF EXISTS (
        SELECT 1
        FROM expected_documents e
        JOIN actual a USING (id, title, content)
        WHERE abs(e.score - a.score) > 1e-6
    ) THEN
        RAISE EXCEPTION 'Test failed: query result differs from expected (hybrid maxsim weighted)!';
    END IF;
    DROP TABLE actual;
    DROP TABLE expected_documents;

    CREATE TEMP TABLE expected_documents (id INTEGER, title TEXT, content TEXT, score REAL);
    INSERT INTO expected_documents VALUES (1, 'Machine Learning Basics', 'Introduction to machine learning algorithms and techniques', 0.5993454),
    (10, 'Reinforcement Learning', 'Exploring agent-based learning and decision-making', 0.06154764),
    (3, 'AI and Neural Networks', 'Deep dive into artificial intelligence and neural networks', 0.059742294),
    (2, 'Data Science Guide', 'Comprehensive guide to data science methodologies', 0.040818404),
    (4, 'Python Programming', 'Learn Python programming from basics to advanced', 0.040320393),
    (5, 'Database Design', 'Principles of database design and optimization', 0.040067032),
    (6, 'Natural Language Processing', 'Study of NLP techniques and applications in AI', 0.039833803),
    (7, 'Cloud Computing', 'Introduction to cloud infrastructure and distributed systems', 0.039623387),
    (8, 'Computer Vision', 'Techniques for image recognition and object detection', 0.039435007),
    (9, 'Big Data Architecture', 'Designing scalable systems for big data processing', 0.039266635);
    CREATE TEMP TABLE actual AS SELECT id, title, content, (embedding_2d, keywords) <#> (ARRAY[[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]], 'machine learning', 0.4, 0.6)::deeplake_hybrid_record_weighted as score FROM documents ORDER BY score LIMIT 10;
    IF EXISTS (
        SELECT 1
        FROM expected_documents e
        JOIN actual a USING (id, title, content)
        WHERE abs(e.score - a.score) > 1e-6
    ) THEN
        RAISE EXCEPTION 'Test failed: query result differs from expected (hybrid maxsim weighted 1)!';
    END IF;
    DROP TABLE actual;
    DROP TABLE expected_documents;
-------------------------------------- End of Hybrid Search with Maxsim and BM25 Similarity --------------------------------

    RAISE NOTICE 'Test passed';
    EXCEPTION
    WHEN OTHERS THEN
      RAISE NOTICE 'ERROR: Test failed: %', SQLERRM;
  END;
  -- Cleanup
  DROP TABLE IF EXISTS documents CASCADE;
  DROP EXTENSION IF EXISTS pg_deeplake CASCADE;
END $$;
