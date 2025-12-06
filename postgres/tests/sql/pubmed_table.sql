\i sql/utils.psql

DROP EXTENSION IF EXISTS pg_deeplake CASCADE;
CREATE EXTENSION pg_deeplake;

DO $$ BEGIN
    DECLARE
    BEGIN
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
        ) USING deeplake WITH (dataset_path='s3://indra-benchmarks/source_datasets/pubmed-15m-noquantized/');
        CREATE INDEX index_for_emb ON pubmed_15m_noquantized USING deeplake_index (embedding DESC);
        SET enable_seqscan = off;

        IF NOT is_using_index_scan(
            'WITH fixed_array AS ('
            '    SELECT generate_random_float_array(1024) AS arr'
            ')'
            'SELECT pmid '
            'FROM pubmed_15m_noquantized, fixed_array '
            'ORDER BY embedding <#> (SELECT arr FROM fixed_array LIMIT 1) '
            'LIMIT 1;') THEN
            RAISE EXCEPTION 'Query must use an index scan!';
        END IF;
    END;

    -- Cleanup
    DROP EXTENSION IF EXISTS pg_deeplake CASCADE;
    RESET enable_seqscan;
    RESET log_min_messages;
    RESET client_min_messages;
END;
$$;
