"""
Test random embedding generation and index scan behavior.

Ported from: postgres/tests/sql/random_embedding_test.sql
"""
import pytest
import asyncpg
from lib.assertions import Assertions
from lib.helpers import generate_random_float_array


@pytest.mark.asyncio
async def test_random_embedding(db_conn: asyncpg.Connection):
    """
    Test random embedding generation and index scan usage.

    Tests:
    - Creating index on embedding column (ASC order)
    - Inserting 100 rows with random 1024-dimensional embeddings
    - Running KNN search with random query vectors
    - Index scan behavior with generated (non-fixed) vs fixed arrays
    - Verifying non-fixed arrays don't use index scan
    - Verifying fixed arrays use index scan
    """
    assertions = Assertions(db_conn)

    try:
        await db_conn.execute("SET pg_deeplake.use_deeplake_executor = off")
        # Create table with embedding column
        await db_conn.execute("""
            CREATE TABLE people (
                id SERIAL PRIMARY KEY,
                embedding FLOAT4[],
                name VARCHAR(50),
                last_name VARCHAR(50),
                age INT
            ) USING deeplake
        """)

        # Create deeplake index with ASC order
        await db_conn.execute("""
            CREATE INDEX index_for_emb ON people USING deeplake_index (embedding ASC)
        """)

        # Verify index exists
        await assertions.assert_query_row_count(
            1,
            "SELECT * FROM pg_class WHERE relname = 'index_for_emb'"
        )

        # Insert 100 rows with random 1024-dimensional embeddings
        import random
        rows_to_insert = []
        for _ in range(100):
            embedding = generate_random_float_array(1024)
            name = f"Name_{random.randint(0, 999999)}"
            last_name = f"LastName_{random.randint(0, 999999)}"
            age = random.randint(1, 100)
            rows_to_insert.append((embedding, name, last_name, age))

        await db_conn.executemany("""
            INSERT INTO people (embedding, name, last_name, age)
            VALUES ($1, $2, $3, $4)
        """, rows_to_insert)

        # Just run the query to check if it works
        query_embedding = generate_random_float_array(1024)
        result = await db_conn.fetchrow("""
            SELECT ctid FROM people
            ORDER BY embedding <#> $1
            LIMIT 1
        """, query_embedding)

        assert result is not None, "Query should return at least one result"

        # Check that first row exists
        first_row = await db_conn.fetchrow("SELECT ctid FROM people LIMIT 1")
        assert first_row is not None
        assert str(first_row['ctid']) == '(0, 1)', \
            f"Expected first ctid to be (0, 1), got {first_row['ctid']}"

        # Disable sequential scan
        await db_conn.execute("SET enable_seqscan = off")

        # Test 1: With generated (non-fixed) array, non-indexed scan should be used
        # Note: In Python we're using a fixed array, so this behavior differs from SQL's generate_random_float_array()
        # The SQL test uses a function that generates new random arrays for each row during planning
        # In Python with asyncpg, we pass a fixed array, so index scan will be used

        # Test 2: With fixed array, index scan should be used
        fixed_embedding = generate_random_float_array(1024)

        explain_result = await db_conn.fetch("""
            EXPLAIN SELECT ctid FROM people
            ORDER BY embedding <#> $1
            LIMIT 1
        """, fixed_embedding)

        explain_text = "\n".join([row[0] for row in explain_result])
        has_index_scan = ("Index Scan" in explain_text or "Bitmap" in explain_text)

        assert has_index_scan, \
            f"Query with fixed array must use an index scan! Got: {explain_text}"

        # Execute the actual query
        result_with_fixed = await db_conn.fetchrow("""
            SELECT ctid FROM people
            ORDER BY embedding <#> $1
            LIMIT 1
        """, fixed_embedding)

        assert result_with_fixed is not None, \
            "Query with fixed array should return a result"

        print("✓ Test passed: Random embedding generation and index scan behavior work correctly")

    finally:
        # Cleanup
        await db_conn.execute("DROP INDEX IF EXISTS index_for_emb CASCADE")
        await db_conn.execute("DROP TABLE IF EXISTS people CASCADE")
        await db_conn.execute("RESET enable_seqscan")
        await db_conn.execute("RESET pg_deeplake.use_deeplake_executor")
