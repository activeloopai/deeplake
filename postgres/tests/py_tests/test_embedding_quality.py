"""
Test embedding quality with KNN searches on high-dimensional vectors.

Ported from: postgres/tests/sql/embedding_quality_test.sql
"""
import pytest
import asyncpg
import numpy as np
import random
from lib.assertions import Assertions
from lib.helpers import generate_random_float_array


@pytest.mark.asyncio
@pytest.mark.slow
async def test_embedding_quality(db_conn: asyncpg.Connection):
    """
    Test embedding quality by verifying KNN search returns exact matches.

    Tests:
    - Creating index on 1024-dimensional embeddings
    - Inserting 2000 rows with random embeddings
    - Running 2000 KNN searches
    - Verifying each embedding is closest to itself
    - Index scan usage verification
    """
    assertions = Assertions(db_conn)

    try:
        # Disable deeplake executor
        await db_conn.execute("SET pg_deeplake.use_deeplake_executor = off")

        # Create table with embedding column
        await db_conn.execute("""
            CREATE TABLE people (
                id SERIAL PRIMARY KEY,
                embedding EMBEDDING,
                name VARCHAR(50),
                last_name VARCHAR(50),
                age INT
            ) USING deeplake
        """)

        # Create deeplake index on embedding column
        await db_conn.execute("""
            CREATE INDEX index_for_emb ON people USING deeplake_index (embedding DESC)
        """)

        # Verify index exists
        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM pg_class WHERE relname = 'index_for_emb'"
        )

        # Insert 2000 rows with random 1024-dimensional embeddings
        # Use generate_series with generate_random_float_array for performance
        # await db_conn.execute("""
        #     INSERT INTO people (embedding, name, last_name, age)
        #     SELECT
        #         generate_random_float_array(1024),
        #         'Name_' || trunc(random() * 1000000),
        #         'LastName_' || trunc(random() * 1000000),
        #         trunc(random() * 100) + 1
        #     FROM generate_series(1, 2000)
        # """)

        # Use asyncpg's executemany for better performance
        #rows_to_insert = []
        #for _ in range(2000):
        #    embedding = generate_random_float_array(1024)
        #    name = f"Name_{random.randint(0, 999999)}"
        #    last_name = f"LastName_{random.randint(0, 999999)}"
        #    age = random.randint(1, 100)
        #    rows_to_insert.append((embedding, name, last_name, age))
       
        #await db_conn.executemany("""
        #    INSERT INTO people (embedding, name, last_name, age)
        #    VALUES ($1, $2, $3, $4)
        #""", rows_to_insert)

        embeddings = np.random.random((2000, 1024)).astype(np.float32)
        records = [
            (embeddings[i].tolist(),
             f"Name_{random.randint(0, 999999)}",
             f"LastName_{random.randint(0, 999999)}",
             random.randint(1, 100))
            for i in range(2000)
        ]

        await db_conn.copy_records_to_table(
            'people',
            records=records,
            columns=['embedding', 'name', 'last_name', 'age']
        )

        # Verify row count
        await assertions.assert_table_row_count(2000, "people")

        # Disable sequential scan to force index usage
        await db_conn.execute("SET enable_seqscan = off")

        # Run KNN quality test
        pass_count = 0
        fail_count = 0

        # Get all rows to test against
        all_rows = await db_conn.fetch("SELECT ctid, embedding FROM people")

        for i, row in enumerate(all_rows, 1):
            original_ctid = row['ctid']
            query_embedding = row['embedding']

            # Verify index scan is used
            explain_result = await db_conn.fetch(
                "EXPLAIN SELECT ctid FROM people ORDER BY embedding <#> $1 LIMIT 1",
                query_embedding
            )
            explain_text = "\n".join([r[0] for r in explain_result])
            has_index_scan = ("Index Scan" in explain_text or "Bitmap" in explain_text)

            if not has_index_scan:
                raise AssertionError(f"Query must use an index scan! Got: {explain_text}")

            # Perform KNN search
            result = await db_conn.fetchrow("""
                SELECT ctid FROM people
                ORDER BY embedding <#> $1
                LIMIT 1
            """, query_embedding)

            result_ctid = result['ctid']

            # Compare results
            if str(original_ctid) != str(result_ctid):
                fail_count += 1
                print(f"Run {i}: ❌ Mismatch - Original CTID: {original_ctid}, KNN result CTID: {result_ctid}")
            else:
                pass_count += 1
                if i <= 10 or i % 100 == 0:  # Print first 10 and every 100th
                    print(f"Run {i}: ✅ Match - Expected CTID: {original_ctid}")

        # Require at least > 50% success rate (1000/2000) to match SQL test expectations
        # With normalized normal distribution, we should achieve this easily
        min_required_passes = 1001
        assert pass_count >= min_required_passes, \
            f"Test failed! Only {pass_count}/{len(all_rows)} runs passed (minimum {min_required_passes} required)."

        print(f"✓ Test passed: Embedding quality test successful! {pass_count}/{len(all_rows)} runs passed.")

    finally:
        # Cleanup
        await db_conn.execute("DROP INDEX IF EXISTS index_for_emb CASCADE")
        await db_conn.execute("DROP TABLE IF EXISTS people CASCADE")
        await db_conn.execute("RESET enable_seqscan")
        await db_conn.execute("RESET pg_deeplake.use_deeplake_executor")
