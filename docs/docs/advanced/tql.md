---
seo_title: "Deep Lake TQL | Tensor Query Language | Query Multi-Modal Data"
description: "Up to 10x More Efficient Data Retrieval with TQL, Deep Lake's Custom Query Language Query Datasets, Join Cross-Cloud Datasets, Manipulate Multi-Modal Datasets."
---

# TQL Syntax

## Overview

Deep Lake offers a performant SQL-based query engine called "TQL" (Tensor Query Language) optimized for machine learning and AI workloads. TQL combines familiar SQL syntax with powerful tensor operations, enabling efficient querying of embeddings, images, and other multi-modal data.

## Basic Usage

### Dataset Queries 

TQL can be used directly on a dataset or across multiple datasets:

<!-- test-context
```python
import numpy as np
import deeplake
from deeplake import types

ds = deeplake.create("tmp://")
ds.add_column("id", "int32")
ds.add_column("column_name", "text")

def get_builtin_signature(func):
    name = func.__name__
    doc = func.__doc__ or ''
    sig = doc.split('\n')[0].strip()
    return f"{name}{sig}"

def open(*args, **kwargs):
    return ds

open.__signature__ = get_builtin_signature(deeplake.open)
deeplake.open = open

def query(*args, **kwargs):
    return ds

query.__signature__ = get_builtin_signature(deeplake.query)
deeplake.query = query
```
-->

```python
# Query on a single dataset (no FROM needed)
ds = deeplake.open("al://org_name/dataset_name")
result = ds.query("SELECT * WHERE id > 10")

# Query across datasets (requires FROM)
result = deeplake.query('SELECT * FROM "al://my_org/dataset_name" WHERE id > 10')
```

### Query Syntax

#### String Values
String literals must use single quotes:
```sql
SELECT * WHERE contains(column_name, 'text_value')
```

#### Special Characters
Column or dataset names with special characters need double quotes:
```sql
SELECT * WHERE contains("column-name", 'text_value')
SELECT * FROM "al://my_org/dataset" WHERE id > 10
```

!!! tip
    When writing queries in Python, remember to properly escape quotes:
    ```python
    # Using escape characters
    query = "SELECT * WHERE contains(\"column-name\", 'text_value')"

    # Using different quote types
    query = 'SELECT * WHERE contains("column-name", \'text_value\')'

    # Using triple quotes
    query = """
        SELECT * WHERE contains("column-name", 'text_value')
    """
    ```

## Vector Operations

### Similarity Search
TQL provides multiple methods for vector similarity search:

```sql
-- Cosine similarity (higher is more similar)
SELECT * 
ORDER BY COSINE_SIMILARITY(embeddings, ARRAY[0.1, 0.2, ...]) DESC
LIMIT 100

-- L2 norm/Euclidean distance (lower is more similar)
SELECT * 
ORDER BY L2_NORM(embeddings - ARRAY[0.1, 0.2, ...]) ASC
LIMIT 100

-- L1 norm/Manhattan distance
SELECT * 
ORDER BY L1_NORM(embeddings - ARRAY[0.1, 0.2, ...]) ASC
LIMIT 100

-- L∞ norm/Chebyshev distance
SELECT * 
ORDER BY LINF_NORM(embeddings - ARRAY[0.1, 0.2, ...]) ASC
LIMIT 100
```

### ColPali MAXSIM Search
TQL supports the MAXSIM operator for efficient similarity search with ColPali embeddings:

```sql
-- Using MAXSIM with ColPali embeddings
SELECT *, MAXSIM(
    document_embeddings,
    ARRAY[
        ARRAY[0.1, 0.2, 0.3],
        ARRAY[0.4, 0.5, 0.6],
        ARRAY[0.7, 0.8, 0.9]
    ]
) AS score
ORDER BY MAXSIM(
    document_embeddings,
    ARRAY[
        ARRAY[0.1, 0.2, 0.3],
        ARRAY[0.4, 0.5, 0.6],
        ARRAY[0.7, 0.8, 0.9]
    ]
) DESC
LIMIT 10
```

### JSON and Dictionary Operations

TQL provides powerful operations for querying JSON and dictionary-like data:

```sql
-- Access JSON fields
SELECT metadata['timestamp'] as time, metadata['author'] as author
WHERE metadata['status'] = 'published'

-- Query nested JSON structures
SELECT * WHERE config['model']['type'] = 'transformer'

-- Array indexing in JSON
SELECT data[0] as first_item, data[-1] as last_item

-- Get all keys from JSON object
SELECT KEYS(metadata) as available_fields

-- Check if key exists
SELECT * WHERE ANY(KEYS(metadata) == 'timestamp')

-- Filter by JSON field values
SELECT * WHERE metadata['score'] > 0.8 AND metadata['verified'] = true
```

!!! note "JSON Access Syntax"
    - Use square brackets `['key']` for string keys
    - Use square brackets `[0]` for array indices
    - Negative indices work for arrays: `[-1]` gets the last element

### Text Search

#### Semantic Search with BM25
```sql
-- Find semantically similar text
SELECT *
ORDER BY BM25_SIMILARITY(text_column, 'search query text') DESC
LIMIT 10
```

#### Keyword and Exact Text Search
```sql
-- Find exact keyword matches (requires Inverted index)
SELECT * WHERE CONTAINS(text_column, 'keyword')

-- Find whole text matches (requires Exact index)  
SELECT * WHERE text_column = 'exact full text match'
```

#### Exact Text Search
```sql
-- Find exact text matches (requires Exact index)
SELECT * WHERE EQUALS(text_column, 'exact text match')
```

### Numeric Queries

Numeric columns with indexes support efficient comparison operations:

```sql
-- Range queries (requires Numeric Inverted index)
SELECT * WHERE numeric_column > 100
SELECT * WHERE numeric_column >= 50 AND numeric_column <= 200
SELECT * WHERE numeric_column BETWEEN 10 AND 50

-- Value list matching
SELECT * WHERE numeric_column IN (10, 20, 30, 40)
```

### Pattern Matching with LIKE

TQL supports SQL-style pattern matching with wildcards:

```sql
-- Using wildcards (* for any characters, ? for single character)
SELECT * WHERE text_column LIKE '*apple*'  -- Contains 'apple'
SELECT * WHERE text_column LIKE 'data?'    -- Matches 'data' followed by one char
SELECT * WHERE filename LIKE '*.jpg'       -- Ends with .jpg

-- Exact match with LIKE
SELECT * WHERE category LIKE 'product'

-- Case-insensitive pattern matching with ILIKE
SELECT * WHERE text_column ILIKE '*APPLE*'  -- Matches 'apple', 'Apple', 'APPLE', etc.
SELECT * WHERE email ILIKE '*@gmail.com'    -- Case-insensitive email domain match
SELECT * WHERE title ILIKE 'chapter?'       -- Matches 'Chapter1', 'chapter2', 'CHAPTER3', etc.
```

!!! note
    - TQL uses `*` for multiple characters and `?` for single character wildcards, similar to shell globbing patterns
    - `LIKE` is case-sensitive, while `ILIKE` is case-insensitive

## Advanced Features

### Cross-Cloud Dataset Joins
TQL enables joining datasets across different cloud storage providers:

```sql
-- Join datasets from different storage providers
SELECT 
    i.image,
    i.embedding,
    m.labels,
    m.metadata
FROM "s3://bucket1/images" AS i
JOIN "gcs://bucket2/metadata" AS m 
    ON i.id = m.image_id
WHERE m.verified = true
ORDER BY COSINE_SIMILARITY(i.embedding, ARRAY[...]) DESC
```

### Virtual Columns
Create computed columns on the fly:

```sql
-- Compute similarity scores
SELECT *,
    COSINE_SIMILARITY(embedding, ARRAY[...]) as similarity_score
FROM dataset
ORDER BY similarity_score DESC

-- Complex computations
SELECT *,
    column_1 + column_3 as sum,
    any(boxes[:,0]) < 0 as box_beyond_image
WHERE label = 'person'
```

### Subqueries

TQL supports nested queries for complex data processing:

```sql
-- Basic subquery
SELECT * FROM (
    SELECT * WHERE labels < 4
) WHERE labels > 2

-- Subquery with computed columns
SELECT * FROM (
    SELECT *, COSINE_SIMILARITY(embedding, data(embedding, 1)) as score
) ORDER BY score DESC LIMIT 10

-- Multiple subqueries with joins
SELECT *
FROM (SELECT * FROM "s3://bucket/dataset1" WHERE verified = true) AS t1
JOIN (SELECT * FROM "gcs://bucket/dataset2" WHERE active = true) AS t2
ON t1.id = t2.user_id
```

### Parameterized Queries

Use placeholders for dynamic query execution:

<!-- test-context
```python
# Note: Parameterized queries are documented but not yet available in the Python API
# The test setup will skip this example
pass
```
-->

```python
# Define parameterized query with ? placeholders
query = "SELECT * WHERE age > ? AND category = ? AND score < ?"

# Execute with parameters
# result = ds.query(query, parameters=[25, 'premium', 0.9])

# Parameterized sampling
# query = "SELECT * SAMPLE BY sum_weight(labels == ?: 10, True: 1) LIMIT ?"
# result = ds.query(query, parameters=[5, 1000])
```

!!! tip "Query Parameters"
    Use `?` as placeholders in your query and pass actual values via the `parameters` argument. This improves performance by allowing query plan reuse.

### Logical Operations

```sql
-- Combining conditions
SELECT * 
WHERE (contains(text, 'machine learning')
    AND confidence > 0.9)
    OR label IN ('cat', 'dog')

-- Array operations
SELECT * 
WHERE any(logical_and(
    bounding_boxes[:,3] > 0.5,
    confidence > 0.8
))
```

### Data Sampling
```sql
-- Weighted random sampling
SELECT * 
SAMPLE BY MAX_WEIGHT(
    high_confidence: 0.7,
    medium_confidence: 0.2,
    low_confidence: 0.1
) LIMIT 1000

-- Sampling with replacement
SELECT * 
SAMPLE BY MAX_WEIGHT(
    positive_samples: 0.5,
    negative_samples: 0.5
) replace True LIMIT 2000
```

### Set Operations

TQL supports combining results from multiple queries:

```sql
-- Union: Combine results from multiple queries
(SELECT * WHERE labels == 1 LIMIT 10)
UNION
(SELECT * WHERE labels == 2 LIMIT 10)

-- Union with different conditions
(SELECT * WHERE category = 'training' LIMIT 1000)
UNION
(SELECT * WHERE category = 'validation' LIMIT 200)
```

### Data Expansion

Expand image or tensor data with optional overlap control:

```sql
-- Expand data by specified dimensions
SELECT * WHERE labels == 0
EXPAND BY 2 2 as expanded_data

-- Expand with overlap enabled
SELECT * WHERE category = 'images'
EXPAND BY 4 4 OVERLAP true as overlapped_tiles

-- Example: Create image patches for training
SELECT image, labels
EXPAND BY 224 224 as patches
WHERE dataset_split = 'train'
```

!!! tip "Image Tiling"
    `EXPAND BY` is particularly useful for creating sliding window patches from large images for computer vision tasks.

### Grouping and Sequences

```sql
-- Group frames into videos
SELECT *
GROUP BY video_id, camera_id

-- Group by multiple columns
SELECT *
GROUP BY category, subcategory

-- Split videos into frames
SELECT *
UNGROUP BY split
```

## Built-in Functions

### Array Operations

- `SHAPE(array)`: Returns array dimensions
  ```sql
  SELECT * WHERE SHAPE(embedding)[0] = 768
  SELECT * WHERE SHAPE(boxes)[0] > 10  -- More than 10 bounding boxes
  ```

- `DATA(column, index)`: Access specific array elements for comparison
  ```sql
  SELECT * ORDER BY L2_NORM(embedding - data(embedding, 10))
  ```

- `NONZERO(array)`: Returns indices of non-zero elements
  ```sql
  -- Returns rows where the first non-zero element is less than 10
  SELECT * WHERE NONZERO(scores)[0] < 10
  ```

### Row Information

- `ROW_NUMBER()`: Returns zero-based row offset
  ```sql
  SELECT *, ROW_NUMBER() WHERE ROW_NUMBER() < 100
  SELECT *, ROW_NUMBER() as id  -- Create virtual ID column
  ```

### Array Logic

- `ANY(condition[, axis])`: True if any element satisfies condition
  ```sql
  SELECT * WHERE ANY(confidence > 0.9)
  SELECT * WHERE ANY(boxes[:,0] < 0)  -- Any box beyond left edge
  ```

- `ALL(condition[, axis])`: True if all elements satisfy condition
  ```sql
  SELECT * WHERE ALL(scores > 0.5)
  SELECT * WHERE ALL(pixels < 255)
  ```

- `ALL_STRICT(condition)`: Stricter version of ALL (returns false for empty arrays)
  ```sql
  SELECT * WHERE ALL_STRICT(values > 0)
  SELECT * WHERE ALL_STRICT(boxes[:,3] > 200)
  ```

- `LOGICAL_AND(array1, array2)`: Element-wise logical AND
  ```sql
  SELECT * WHERE ANY(LOGICAL_AND(confidence > 0.8, area > 100))
  ```

- `LOGICAL_OR(array1, array2)`: Element-wise logical OR
  ```sql
  SELECT * WHERE ANY(LOGICAL_OR(type == 'car', type == 'truck'))
  ```

### Aggregations

- `SUM(array)`: Sum of all elements
  ```sql
  SELECT SUM(prices) as total WHERE category = 'electronics'
  ```

- `AVG(array)`: Average of all elements
  ```sql
  SELECT category, AVG(scores) as avg_score GROUP BY category
  ```

- `PROD(array)`: Product of all elements
  ```sql
  SELECT PROD(dimensions) as volume
  ```

- `AMIN(array)`: Minimum value
  ```sql
  SELECT AMIN(confidence) WHERE label = 'person'
  ```

- `AMAX(array)`: Maximum value
  ```sql
  SELECT AMAX(confidence) WHERE label = 'person'
  ```

### Mathematical Functions

- `SQRT(value)`: Square root
  ```sql
  SELECT * WHERE SQRT(area) > 10
  ```

- `ABS(value)`: Absolute value
  ```sql
  SELECT * WHERE ABS(temperature - 20) < 5
  ```

### Vector Distance Functions

- `DOT(vector1, vector2)`: Dot product of two vectors
  ```sql
  SELECT *, DOT(embedding, ARRAY[0.1, 0.2, ...]) as dot_score
  ORDER BY dot_score DESC
  ```

- `HAMMING_DISTANCE(vector1, vector2)`: Hamming distance between vectors
  ```sql
  SELECT * ORDER BY HAMMING_DISTANCE(binary_vector, ARRAY[0, 1, 1, 0]) ASC
  ```

### JSON/Dictionary Operations

- `KEYS(json_object)`: Returns keys from JSON object
  ```sql
  SELECT KEYS(metadata) as available_fields
  SELECT * WHERE ANY(KEYS(metadata) == 'timestamp')
  ```

### Random Numbers

- `RANDOM()`: Returns a random 32-bit integer number
  ```sql
  -- Shuffle the dataset
  SELECT * ORDER BY RANDOM()

  -- Random sampling with ordering
  SELECT * WHERE RANDOM() % 10 == 0  -- ~10% random sample
  ```

## Custom Functions

TQL supports registering custom Python functions:

```python
# Define and register custom function
def custom_square(a):
    return a * a

deeplake.tql.register_function(custom_square)

# Use in query
results = ds.query("SELECT * WHERE custom_square(column_name) > 10")
```

Custom functions must:

  - Accept numpy arrays as input
  - Return numpy arrays as output
  - Be registered before use in queries

## Query Syntax Reference

### Operators

#### Arithmetic Operators
```sql
SELECT column_1 + column_2 as sum
SELECT column_1 - column_2 as diff
SELECT column_1 * column_2 as product
SELECT column_1 / column_2 as ratio
SELECT value % 10 as modulo
```

#### Comparison Operators
```sql
SELECT * WHERE value == 5    -- or value = 5
SELECT * WHERE value != 5
SELECT * WHERE value > 5
SELECT * WHERE value >= 5
SELECT * WHERE value < 5
SELECT * WHERE value <= 5
SELECT * WHERE value BETWEEN 1 AND 10
SELECT * WHERE value IN (1, 2, 3, 4, 5)
```

#### Logical Operators
```sql
SELECT * WHERE condition1 AND condition2
SELECT * WHERE condition1 OR condition2
SELECT * WHERE NOT condition
```

### Array Indexing and Slicing

```sql
-- Single element access
SELECT column[0] as first_element
SELECT column[-1] as last_element

-- Slicing (Python-style)
SELECT column[0:10] as first_ten
SELECT column[10:] as from_tenth
SELECT column[:10] as up_to_tenth
SELECT column[::2] as every_other

-- Multi-dimensional arrays
SELECT boxes[:,0] as x_coords
SELECT matrix[0:5, 0:3] as submatrix
SELECT * WHERE any(boxes[:,3] > 200)  -- Height > 200
```

### Join Types

```sql
-- Inner join (only matching rows)
SELECT * FROM table1 INNER JOIN table2 ON table1.id = table2.id

-- Left join (all from left, matching from right)
SELECT * FROM table1 LEFT JOIN table2 ON table1.id = table2.id

-- Right join (all from right, matching from left)
SELECT * FROM table1 RIGHT JOIN table2 ON table1.id = table2.id

-- Full join (all from both)
SELECT * FROM table1 FULL JOIN table2 ON table1.id = table2.id

-- Cross join (cartesian product)
SELECT * FROM table1 CROSS JOIN table2

-- Using clause (when column names match)
SELECT * FROM table1 JOIN table2 USING(id)
```

### Query Modifiers

```sql
-- Limit results
SELECT * LIMIT 100
SELECT * LIMIT 0.1 PERCENT  -- Get 0.1% of results

-- Order results
SELECT * ORDER BY column ASC
SELECT * ORDER BY column DESC
SELECT * ORDER BY column1 ASC, column2 DESC

-- Sampling
SELECT * SAMPLE BY 0.1  -- 10% random sample
SELECT * SAMPLE BY labels  -- Sample by label distribution
SELECT * SAMPLE BY labels REPLACE true  -- With replacement
```

## Index Creation for Optimal Performance

### Text Indexes

Create indexes on text columns to enable efficient text search:

```python
# Create column with Inverted index for keyword search
ds.add_column("description", deeplake.types.Text(deeplake.types.Inverted))

# Create column with BM25 index for semantic search
ds.add_column("content", deeplake.types.Text(deeplake.types.BM25))

# Add both Inverted and BM25 indexes to different columns
ds.add_column("document", deeplake.types.Text(deeplake.types.Inverted))
ds.add_column("article", deeplake.types.Text())
ds["article"].create_index(deeplake.types.TextIndex(deeplake.types.BM25))

# Create index on new Text column
ds.add_column("existing_text", deeplake.types.Text())
ds["existing_text"].create_index(deeplake.types.TextIndex(deeplake.types.Inverted))
```

Usage with queries:

```sql
-- Keyword search (requires Inverted index)
SELECT * WHERE CONTAINS(description, 'machine learning')

-- Semantic search (requires BM25 index)
SELECT * ORDER BY BM25_SIMILARITY(content, 'deep learning tutorial') DESC LIMIT 10

-- Pattern matching (requires Inverted index)
SELECT * WHERE description LIKE '*neural network*'
```

### Numeric Indexes

Create inverted indexes on numeric columns for efficient range queries:

```python
# Create column with Numeric Inverted index
ds.add_column("age", deeplake.types.UInt64())
ds["age"].create_index(
    deeplake.types.NumericIndex(deeplake.types.Inverted)
)

# Or on existing column - first create the column
ds.add_column("price", deeplake.types.Float64())
ds["price"].create_index(
    deeplake.types.NumericIndex(deeplake.types.Inverted)
)
```

Usage with queries:

```sql
-- Range queries (requires Numeric Inverted index)
SELECT * WHERE age BETWEEN 18 AND 65
SELECT * WHERE price > 100 AND price <= 1000

-- Value list matching
SELECT * WHERE category_id IN (1, 3, 5, 7, 9)
```

### Embedding Indexes

For vector similarity search on embeddings:

```python
# Create Embedding column with Clustered index (default)
ds.add_column("embedding", deeplake.types.Embedding(size=768))

# Create with Quantized index for faster search with larger datasets
ds.add_column(
    "embedding_quantized",
    deeplake.types.Embedding(
        size=768,
        index_type=deeplake.types.EmbeddingIndex(deeplake.types.ClusteredQuantized)
    )
)

# Add Clustered index to existing Array column - first create the column
ds.add_column("vectors", deeplake.types.Array(dtype=deeplake.types.Float32(), shape=(128,)))
ds["vectors"].create_index(
    deeplake.types.EmbeddingIndex(deeplake.types.Clustered)
)
```

Usage with queries:

```sql
-- Cosine similarity search
SELECT * ORDER BY COSINE_SIMILARITY(embedding, ARRAY[...]) DESC LIMIT 10

-- Combined filtering and similarity search
SELECT * WHERE category = 'products'
ORDER BY COSINE_SIMILARITY(embedding, ARRAY[...]) DESC
LIMIT 20
```

### Managing Indexes

```python
# Check existing indexes
print(ds["column_name"].indexes)

# Drop an index - use columns we created earlier
ds["article"].drop_index(deeplake.types.TextIndex(deeplake.types.BM25))
ds["age"].drop_index(
    deeplake.types.NumericIndex(deeplake.types.Inverted)
)

# Commit changes
ds.commit()
```

!!! tip "Index Best Practices"
    - **Inverted indexes** are essential for `CONTAINS`, `IN` `LIKE`, and range queries
    - **BM25 indexes** enable semantic text search via `BM25_SIMILARITY`
    - Create indexes before ingesting large amounts of data when possible
    - Use `ClusteredQuantized` for embeddings when dataset size > 10M rows
    - Indexes are automatically maintained as you add/update data
