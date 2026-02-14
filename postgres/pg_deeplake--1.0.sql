\echo Use "CREATE EXTENSION pg_deeplake" to load this file.

-- Create custom domain types
-- IMAGE domain: behaves like BYTEA but semantically represents image data
CREATE DOMAIN IMAGE AS BYTEA;
COMMENT ON DOMAIN IMAGE IS 'Binary image data stored as BYTEA';

-- VIDEO domain: behaves like BYTEA but semantically represents video data
CREATE DOMAIN VIDEO AS BYTEA;
COMMENT ON DOMAIN VIDEO IS 'Binary video data stored as BYTEA';

-- FILE domain: behaves like BYTEA but semantically represents file data
CREATE DOMAIN FILE AS BYTEA;
COMMENT ON DOMAIN FILE IS 'Binary file data stored as BYTEA';

-- FILE_ID domain: UUID alias for file identifiers
CREATE DOMAIN FILE_ID AS UUID;
COMMENT ON DOMAIN FILE_ID IS 'UUID identifier for files';

CREATE FUNCTION handle_index_creation() RETURNS event_trigger AS 'pg_deeplake' LANGUAGE C VOLATILE;

-- Create the event trigger to listen for CREATE INDEX events
CREATE EVENT TRIGGER index_creation_trigger
ON ddl_command_start
WHEN TAG IN ('CREATE INDEX')
EXECUTE FUNCTION handle_index_creation();

DROP TABLE IF EXISTS public.pg_deeplake_tables CASCADE;
CREATE TABLE public.pg_deeplake_tables (
    id SERIAL PRIMARY KEY,
    table_oid OID NOT NULL UNIQUE,
    table_name NAME NOT NULL UNIQUE,
    ds_path TEXT NOT NULL UNIQUE
);
COMMENT ON TABLE public.pg_deeplake_tables IS 'Stores table metadata for DeepLake tables';
GRANT SELECT, INSERT, UPDATE, DELETE ON public.pg_deeplake_tables TO PUBLIC;
GRANT USAGE, SELECT, UPDATE ON SEQUENCE public.pg_deeplake_tables_id_seq TO PUBLIC;

DROP TABLE IF EXISTS public.pg_deeplake_views CASCADE;
CREATE TABLE public.pg_deeplake_views (
    view_name NAME NOT NULL PRIMARY KEY,
    query_string TEXT NOT NULL
);
COMMENT ON TABLE public.pg_deeplake_views IS 'Stores view definitions for views on DeepLake tables';
GRANT SELECT, INSERT, UPDATE, DELETE ON public.pg_deeplake_views TO PUBLIC;

DROP TABLE IF EXISTS public.pg_deeplake_metadata CASCADE;
CREATE TABLE public.pg_deeplake_metadata (
    id SERIAL PRIMARY KEY,
    table_name NAME NOT NULL,
    column_name TEXT NOT NULL,
    index_name NAME,
    index_type NAME,
    order_type int,
    index_id oid,
    UNIQUE(table_name, column_name)
);
COMMENT ON TABLE public.pg_deeplake_metadata IS 'Stores index metadata for DeepLake indexes';
GRANT SELECT, INSERT, UPDATE, DELETE ON public.pg_deeplake_metadata TO PUBLIC;
GRANT USAGE, SELECT, UPDATE ON SEQUENCE public.pg_deeplake_metadata_id_seq TO PUBLIC;

CREATE FUNCTION create_deeplake_table(tablename TEXT, path TEXT)
RETURNS void
AS 'pg_deeplake', 'create_deeplake_table'
LANGUAGE C;
COMMENT ON FUNCTION create_deeplake_table(tablename TEXT, path TEXT) IS
'Creates a Postgres table with the specified name on top of DeepLake dataset in specified path';

-- float4[] functions
CREATE FUNCTION deeplake_cosine_similarity(float4[], float4[]) RETURNS float4
	AS 'pg_deeplake' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION deeplake_vector_lt(float4[], float4[]) RETURNS bool
	AS 'pg_deeplake' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION deeplake_vector_le(float4[], float4[]) RETURNS bool
	AS 'pg_deeplake' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION deeplake_vector_eq(float4[], float4[]) RETURNS bool
	AS 'pg_deeplake' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION deeplake_vector_ne(float4[], float4[]) RETURNS bool
	AS 'pg_deeplake' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION deeplake_vector_ge(float4[], float4[]) RETURNS bool
	AS 'pg_deeplake' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION deeplake_vector_gt(float4[], float4[]) RETURNS bool
	AS 'pg_deeplake' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION deeplake_vector_compare(float4[], float4[]) RETURNS int4
	AS 'pg_deeplake' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION deeplake_maxsim(float4[][], float4[][]) RETURNS float4
	AS 'pg_deeplake' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

-- float4[] operators

CREATE OPERATOR <#> (
	LEFTARG = float4[], RIGHTARG = float4[], PROCEDURE = deeplake_cosine_similarity,
	COMMUTATOR = '<#>'
);

CREATE OPERATOR < (
	LEFTARG = float4[], RIGHTARG = float4[], PROCEDURE = deeplake_vector_lt,
	COMMUTATOR = > , NEGATOR = >= ,
	RESTRICT = scalarltsel, JOIN = scalarltjoinsel
);

CREATE OPERATOR <= (
	LEFTARG = float4[], RIGHTARG = float4[], PROCEDURE = deeplake_vector_le,
	COMMUTATOR = >= , NEGATOR = > ,
	RESTRICT = scalarlesel, JOIN = scalarlejoinsel
);

CREATE OPERATOR = (
	LEFTARG = float4[], RIGHTARG = float4[], PROCEDURE = deeplake_vector_eq,
	COMMUTATOR = = , NEGATOR = <> ,
	RESTRICT = eqsel, JOIN = eqjoinsel
);

CREATE OPERATOR <> (
	LEFTARG = float4[], RIGHTARG = float4[], PROCEDURE = deeplake_vector_ne,
	COMMUTATOR = <> , NEGATOR = = ,
	RESTRICT = eqsel, JOIN = eqjoinsel
);

CREATE OPERATOR >= (
	LEFTARG = float4[], RIGHTARG = float4[], PROCEDURE = deeplake_vector_ge,
	COMMUTATOR = <= , NEGATOR = < ,
	RESTRICT = scalargesel, JOIN = scalargejoinsel
);

CREATE OPERATOR > (
	LEFTARG = float4[], RIGHTARG = float4[], PROCEDURE = deeplake_vector_gt,
	COMMUTATOR = < , NEGATOR = <= ,
	RESTRICT = scalargtsel, JOIN = scalargtjoinsel
);

-- Handler function declaration
CREATE FUNCTION deeplake_index_handler(INTERNAL) RETURNS index_am_handler AS 'pg_deeplake' LANGUAGE C VOLATILE;

-- Define access method
CREATE ACCESS METHOD deeplake_index TYPE INDEX HANDLER deeplake_index_handler;

COMMENT ON ACCESS METHOD deeplake_index IS 'DeepLake index access method';

CREATE OPERATOR CLASS deeplake_vector_ops
    DEFAULT FOR TYPE float4[] USING deeplake_index AS
    OPERATOR 1 < (float4[], float4[]),
    OPERATOR 2 <= (float4[], float4[]),
    OPERATOR 3 = (float4[], float4[]),
    OPERATOR 4 >= (float4[], float4[]),
    OPERATOR 5 > (float4[], float4[]),
    OPERATOR 6 <#> (float4[], float4[]) FOR ORDER BY float_ops,
    FUNCTION 1 deeplake_vector_compare(float4[], float4[]);

-- BM25 similarity for text
CREATE FUNCTION deeplake_bm25_similarity(text, text) RETURNS float4
    AS 'pg_deeplake', 'deeplake_bm25_similarity_text'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

-- BM25 similarity operator for text
CREATE OPERATOR <#> (
    LEFTARG = text, RIGHTARG = text, PROCEDURE = deeplake_bm25_similarity,
    COMMUTATOR = '<#>'
);

-- Text contains function for full-text search
CREATE FUNCTION deeplake_text_contains(text, text) RETURNS boolean
    AS 'pg_deeplake', 'deeplake_text_contains'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION contains(col text, search text) RETURNS boolean
AS $$
    SELECT col @> search
$$ LANGUAGE SQL IMMUTABLE STRICT PARALLEL SAFE;

-- Text contains operator @>
CREATE OPERATOR @> (
    LEFTARG = text, RIGHTARG = text, PROCEDURE = deeplake_text_contains,
    COMMUTATOR = '@>',
    RESTRICT = contsel,
    JOIN = contjoinsel
);

CREATE FUNCTION deeplake_bm25_cmp(text, text) RETURNS int4
    AS $$
    DECLARE
        sim1 float4;
        sim2 float4;
    BEGIN
        sim1 := deeplake_bm25_similarity($1, $2);
        sim2 := deeplake_bm25_similarity($2, $1);
        RETURN CASE
            WHEN sim1 > sim2 THEN -1  -- First text more similar
            WHEN sim1 < sim2 THEN 1   -- Second text more similar
            ELSE 0                    -- Equal similarity
        END;
    END;
    $$ LANGUAGE plpgsql IMMUTABLE STRICT PARALLEL SAFE;

-- Unified operator class for text supporting BM25, exact match, and contains
-- The backend will use the appropriate algorithm based on the index_type parameter
-- Strategy numbers: 3 for =, 7 for @>, 1 for <#> (ORDER BY)
CREATE OPERATOR CLASS deeplake_text_ops
    DEFAULT FOR TYPE text USING deeplake_index AS
    OPERATOR 3 = (text, text),
    OPERATOR 7 @> (text, text),
    OPERATOR 1 <#> (text, text) FOR ORDER BY float_ops,
    FUNCTION 1 deeplake_bm25_cmp(text, text);

CREATE OPERATOR CLASS deeplake_uuid_ops
    DEFAULT FOR TYPE uuid USING deeplake_index AS
    OPERATOR 3 = (uuid, uuid);

-- =============================================
-- HYBRID SEARCH INTERFACE
-- =============================================

CREATE FUNCTION array_ndims(anyarray) RETURNS int
LANGUAGE sql IMMUTABLE STRICT
AS $$
    SELECT coalesce(array_upper($1, 1), 0)
$$;

CREATE DOMAIN EMBEDDING AS real[] CHECK (array_ndims(VALUE) = 1);
COMMENT ON DOMAIN EMBEDDING IS 'Embedding vector stored as 1D float4 array';
CREATE DOMAIN EMBEDDING_2D AS real[][] CHECK (array_ndims(VALUE) = 2);
COMMENT ON DOMAIN EMBEDDING_2D IS 'Embedding vectors stored as 2D float4 array';

CREATE TYPE deeplake_hybrid_record AS (
    embedding float4[],
    text_value text
);

CREATE TYPE deeplake_hybrid_record_weighted AS (
    embedding float4[],
    text_value text,
    embedding_weight float8,
    text_weight float8
);

CREATE FUNCTION deeplake_hybrid_record_to_weighted(deeplake_hybrid_record)
RETURNS deeplake_hybrid_record_weighted AS $$
BEGIN
    RETURN ($1.embedding, $1.text_value, 1.0, 1.0)::deeplake_hybrid_record_weighted;
END;
$$ LANGUAGE plpgsql IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION deeplake_hybrid_record_weighted_to_simple(deeplake_hybrid_record_weighted)
RETURNS deeplake_hybrid_record AS $$
BEGIN
    RETURN ($1.embedding, $1.text_value)::deeplake_hybrid_record;
END;
$$ LANGUAGE plpgsql IMMUTABLE STRICT PARALLEL SAFE;

CREATE CAST (deeplake_hybrid_record AS deeplake_hybrid_record_weighted) 
WITH FUNCTION deeplake_hybrid_record_to_weighted(deeplake_hybrid_record) 
AS IMPLICIT;

CREATE CAST (deeplake_hybrid_record_weighted AS deeplake_hybrid_record) 
WITH FUNCTION deeplake_hybrid_record_weighted_to_simple(deeplake_hybrid_record_weighted) 
AS IMPLICIT;

CREATE FUNCTION deeplake_hybrid_record(embedding float4[], text_value text)
RETURNS deeplake_hybrid_record_weighted AS $$
BEGIN
    RETURN ($1, $2, 0.5, 0.5)::deeplake_hybrid_record_weighted;
END;
$$ LANGUAGE plpgsql IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION deeplake_hybrid_record(embedding float4[], text_value text, embedding_weight float8, text_weight float8)
RETURNS deeplake_hybrid_record_weighted AS $$
BEGIN
    RETURN ($1, $2, $3, $4)::deeplake_hybrid_record_weighted;
END;
$$ LANGUAGE plpgsql IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION deeplake_hybrid_search(deeplake_hybrid_record, deeplake_hybrid_record_weighted)
RETURNS float4
AS 'pg_deeplake', 'deeplake_hybrid_search'
LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE OPERATOR <#> (
    LEFTARG = deeplake_hybrid_record,
    RIGHTARG = deeplake_hybrid_record_weighted,
    PROCEDURE = deeplake_hybrid_search,
    COMMUTATOR = '<#>'
);

CREATE FUNCTION deeplake_hybrid_cmp(deeplake_hybrid_record, deeplake_hybrid_record_weighted)
RETURNS int4
    AS $$
    DECLARE
        sim1 float4;
        sim2 float4;
    BEGIN
        sim1 := deeplake_hybrid_search($1, $2);
        sim2 := deeplake_hybrid_search($2, $1);
        RETURN CASE
            WHEN sim1 > sim2 THEN -1  -- First record more similar
            WHEN sim1 < sim2 THEN 1   -- Second record more similar
            ELSE 0                    -- Equal similarity
        END;
    END;
    $$ LANGUAGE plpgsql IMMUTABLE STRICT PARALLEL SAFE;


-- operator class for hybrid search
CREATE OPERATOR CLASS deeplake_hybrid_ops
    DEFAULT FOR TYPE deeplake_hybrid_record USING deeplake_index AS
    OPERATOR 1 <#> (deeplake_hybrid_record, deeplake_hybrid_record_weighted) FOR ORDER BY float_ops,
    FUNCTION 1 deeplake_hybrid_cmp(deeplake_hybrid_record, deeplake_hybrid_record_weighted);

-- operator classes for inverted index
-- integer
CREATE OPERATOR CLASS deeplake_inverted_index_ops_int
    DEFAULT FOR TYPE integer USING deeplake_index AS
    OPERATOR 1 < (integer, integer),
    OPERATOR 2 <= (integer, integer),
    OPERATOR 3 = (integer, integer),
    OPERATOR 4 >= (integer, integer),
    OPERATOR 5 > (integer, integer);

-- bigint
CREATE OPERATOR CLASS deeplake_inverted_index_ops_bigint
    DEFAULT FOR TYPE bigint USING deeplake_index AS
    OPERATOR 1 < (bigint, bigint),
    OPERATOR 2 <= (bigint, bigint),
    OPERATOR 3 = (bigint, bigint),
    OPERATOR 4 >= (bigint, bigint),
    OPERATOR 5 > (bigint, bigint);

-- smallint
CREATE OPERATOR CLASS deeplake_inverted_index_ops_smallint
    DEFAULT FOR TYPE smallint USING deeplake_index AS
    OPERATOR 1 < (smallint, smallint),
    OPERATOR 2 <= (smallint, smallint),
    OPERATOR 3 = (smallint, smallint),
    OPERATOR 4 >= (smallint, smallint),
    OPERATOR 5 > (smallint, smallint);

-- numeric / decimal
CREATE OPERATOR CLASS deeplake_inverted_index_ops_numeric
    DEFAULT FOR TYPE numeric USING deeplake_index AS
    OPERATOR 1 < (numeric, numeric),
    OPERATOR 2 <= (numeric, numeric),
    OPERATOR 3 = (numeric, numeric),
    OPERATOR 4 >= (numeric, numeric),
    OPERATOR 5 > (numeric, numeric);

-- real / float4
CREATE OPERATOR CLASS deeplake_inverted_index_ops_real
    DEFAULT FOR TYPE real USING deeplake_index AS
    OPERATOR 1 < (real, real),
    OPERATOR 2 <= (real, real),
    OPERATOR 3 = (real, real),
    OPERATOR 4 >= (real, real),
    OPERATOR 5 > (real, real);

-- double precision / float8
CREATE OPERATOR CLASS deeplake_inverted_index_ops_double
    DEFAULT FOR TYPE double precision USING deeplake_index AS
    OPERATOR 1 < (double precision, double precision),
    OPERATOR 2 <= (double precision, double precision),
    OPERATOR 3 = (double precision, double precision),
    OPERATOR 4 >= (double precision, double precision),
    OPERATOR 5 > (double precision, double precision);

-- Index-aware JSONB field equality function
-- This function signals to the index that it should search for a specific field value
CREATE FUNCTION jsonb_field_eq(col jsonb, field text, value text) RETURNS boolean
AS 'pg_deeplake', 'deeplake_jsonb_field_eq'
LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

COMMENT ON FUNCTION jsonb_field_eq(jsonb, text, text) IS
'Index-friendly JSONB field equality. Transformed automatically from ->> expressions.';

-- JSONB operator class for inverted index
-- Uses built-in @> containment operator for field search (strategy 7)
-- Note: Our backend extracts the field value and uses inverted text index
CREATE OPERATOR CLASS deeplake_jsonb_ops
    DEFAULT FOR TYPE jsonb USING deeplake_index AS
    OPERATOR 7 @> (jsonb, jsonb) FOR SEARCH,
    FUNCTION 1 bttextcmp(text, text);

-- Register the table access method handler function
CREATE FUNCTION deeplake_tableam_handler(internal)
RETURNS table_am_handler
AS 'pg_deeplake', 'deeplake_tableam_handler'
LANGUAGE C STRICT;

-- Create the table access method
CREATE ACCESS METHOD deeplake TYPE TABLE HANDLER deeplake_tableam_handler;
