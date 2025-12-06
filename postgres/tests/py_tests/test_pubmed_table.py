"""
Test external dataset connection (PubMed 15M) (DISABLED - test is disabled in Makefile).

Ported from: postgres/tests/sql/pubmed_table.sql
"""
import pytest
import asyncpg
from lib.assertions import Assertions
from lib.helpers import generate_random_float_array


@pytest.mark.asyncio
@pytest.mark.disabled
async def test_pubmed_table(db_conn: asyncpg.Connection):
    """
    Test connecting to external PubMed 15M dataset and running KNN queries.

    NOTE: This test is disabled by default as it's disabled in the Makefile.
    It requires access to an external Azure dataset.
    Run with: pytest -m disabled

    Tests:
    - Creating table with external dataset_path (Azure blob storage)
    - Creating deeplake index on 1024-dimensional embeddings
    - Running KNN similarity search
    - Index scan usage verification
    """
    assertions = Assertions(db_conn)

    try:
        # Create table connected to external PubMed dataset
        await db_conn.execute("""
            CREATE TABLE IF NOT EXISTS pubmed_15m_noquantized (
                embedding FLOAT4[1024],
                pmid BIGINT,
                title TEXT,
                abstract TEXT,
                keywords TEXT,
                year TEXT,
                publication_month TEXT,
                publication_date BIGINT,
                authors TEXT,
                mesh_terms TEXT,
                text TEXT
            ) USING deeplake
            WITH (dataset_path='az://testactiveloop/indra-benchmarks/source_datasets/pubmed-15m-noquantized/')
        """)

        # Create deeplake index on embedding column
        await db_conn.execute("""
            CREATE INDEX index_for_emb ON pubmed_15m_noquantized
            USING deeplake_index (embedding DESC)
        """)

        # Disable sequential scan to force index usage
        await db_conn.execute("SET enable_seqscan = off")

        # Generate a random query embedding
        query_embedding = generate_random_float_array(1024)

        # Verify index scan is used for KNN search
        explain_result = await db_conn.fetch("""
            EXPLAIN SELECT pmid
            FROM pubmed_15m_noquantized
            ORDER BY embedding <#> $1
            LIMIT 1
        """, query_embedding)

        explain_text = "\n".join([row[0] for row in explain_result])
        has_index_scan = ("Index Scan" in explain_text)

        assert has_index_scan, \
            f"Query must use an index scan! Got: {explain_text}"

        # Run the actual KNN query
        result = await db_conn.fetchrow("""
            SELECT pmid
            FROM pubmed_15m_noquantized
            ORDER BY embedding <#> $1
            LIMIT 1
        """, query_embedding)

        count_result = await db_conn.fetchrow("""
            SELECT COUNT(*) AS cnt FROM pubmed_15m_noquantized
        """)
        print(f"Total rows in pubmed_15m_noquantized: {count_result['cnt']}")
        if count_result['cnt'] > 0:
            assert result is not None, "Query should return at least one result"
            assert result['pmid'] is not None, "Result should have a pmid"

        print("✓ Test passed: PubMed external dataset connection and KNN search work correctly")

    finally:
        # Cleanup
        await db_conn.execute("DROP TABLE IF EXISTS pubmed_15m_noquantized CASCADE")
        await db_conn.execute("RESET enable_seqscan")
