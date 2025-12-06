# pg_deeplake

PostgreSQL extension for vector similarity search, full-text search, and hybrid search using DeepLake.

## Quick Start

### Installation

```bash
# Pull and run the Docker container
docker run -d \
  --name pg-deeplake \
  -e POSTGRES_PASSWORD=postgres \
  -p 5432:5432 \
  quay.io/activeloopai/pg-deeplake:18
```

### Connect

```bash
# Connect using psql
psql -h localhost -p 5432 -U postgres
```

### Basic Usage

```sql
-- 1. Enable extension
CREATE EXTENSION pg_deeplake;

-- 2. Create a table with DeepLake storage
CREATE TABLE vectors (
    id SERIAL PRIMARY KEY,
    v1 float4[],
    v2 float4[]
) USING deeplake;

-- 3. Create an index
CREATE INDEX index_for_v1 ON vectors USING deeplake_index (v1 DESC);

-- 4. Insert data
INSERT INTO vectors (v1, v2) VALUES
    (ARRAY[1.0, 2.0, 3.0], ARRAY[1.0, 2.0, 3.0]),
    (ARRAY[4.0, 5.0, 6.0], ARRAY[4.0, 5.0, 6.0]),
    (ARRAY[7.0, 8.0, 9.0], ARRAY[7.0, 8.0, 9.0]);

-- 5. Query with cosine similarity
SELECT id, v1 <#> ARRAY[1.0, 2.0, 3.0] AS score
FROM vectors
ORDER BY score DESC
LIMIT 10;
```

### What You Get

- **Vector similarity search** using `<#>` operator (cosine similarity)
- **Full-text search** with BM25 scoring on text columns
- **Hybrid search** combining vectors and text
- **Numeric indexing** for integer/float columns
- **JSON indexing** for JSONB fields
