---
seo_title: "Deep Lake Query API | Efficient Multi-Modal Search"
description: "Deep Lake Query Engine Documentation and Examples."
toc_depth: 2
---
# Query

Deep Lake provides powerful query capabilities through its Tensor Query Language (TQL), with special focus on vector similarity search, text search, and operations on multidimensional arrays.


## Query Functions

::: deeplake.query
    options:
        heading_level: 3

::: deeplake.query_async
    options:
        heading_level: 3

::: deeplake.prepare_query
    options:
        heading_level: 3

::: deeplake.explain_query
    options:
        heading_level: 3

## Query Classes

### Executor

Prepared query execution object.

::: deeplake.Executor
    options:
        heading_level: 4
        members:
            - get_query_string
            - run_single
            - run_single_async
            - run_batch
            - run_batch_async

### ExplainQueryResult

Query explanation and analysis result.

::: deeplake.ExplainQueryResult
    options:
        heading_level: 4
        members:
            - to_dict

## Vector Search

Search by vector similarity:

<!-- test-context
```python
import numpy as np
import deeplake
from deeplake import types

ds = deeplake.create("tmp://")
ds.add_column("embeddings", deeplake.types.Embedding(768))
ds.add_column("features", deeplake.types.Array("int32", 2))
ds.add_column("images", deeplake.types.Image())
ds.add_column("labels", deeplake.types.ClassLabel("int32"))
ds.add_column("species", deeplake.types.ClassLabel("int32"))
ds.add_column("weight", "float32")
ds["species"].metadata["class_names"] = ["cat", "dog", "bird"]

class FutureMock:
    def __init__(self):
        pass

    def result(self):
        return ds

    def is_completed(self):
        return True

class ExecutorMock:
    def __init__(self):
        pass
    
    def get_query_string(self):
        return "SELECT * FROM dataset WHERE condition"
    
    def run_single(self, params=None):
        return ds
    
    def run_single_async(self, params=None):
        return FutureMock()
    
    def run_batch(self, param_list):
        return [ds for _ in param_list]
    
    def run_batch_async(self, param_list):
        return FutureMock()

class ExplainQueryResultMock:
    def __init__(self):
        pass
    
    def to_dict(self):
        return {"execution_plan": "optimized", "index_used": True}
    
    def __str__(self):
        return "Query execution plan: optimized with index usage"

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

def query_async(*args, **kwargs):
    return FutureMock()

query_async.__signature__ = get_builtin_signature(deeplake.query_async)
deeplake.query_async = query_async

def prepare_query(*args, **kwargs):
    return ExecutorMock()

prepare_query.__signature__ = get_builtin_signature(deeplake.prepare_query)
deeplake.prepare_query = prepare_query

def explain_query(*args, **kwargs):
    return ExplainQueryResultMock()

explain_query.__signature__ = get_builtin_signature(deeplake.explain_query)
deeplake.explain_query = explain_query

search_vector = np.random.rand(768)
```
-->

```python
# Cosine similarity search
text_vector = ','.join(str(x) for x in search_vector)
results = deeplake.query(f"""
    SELECT *
    FROM "s3://bucket/embeddings"
    ORDER BY COSINE_SIMILARITY(embeddings, ARRAY[{text_vector}]) DESC
    LIMIT 100
""")
```

## Text Search

Text search using BM25 or keyword matching:

```python
# Semantic search using BM25
results = deeplake.query("""
    SELECT *
    FROM "s3://bucket/documents"
    ORDER BY BM25_SIMILARITY(text, 'search query') DESC
    LIMIT 10
""")

# Keyword search using CONTAINS
results = deeplake.query("""
    SELECT *
    FROM "s3://bucket/metadata"
    WHERE CONTAINS(keywords, 'specific term')
""")
```

## Array Operations

Operate on multidimensional arrays:

```python
# Select specific array dimensions
results = deeplake.query("""
    SELECT features[:, 0:10]
    FROM "s3://bucket/features"
""")

# Filter by array values
results = deeplake.query("""
    SELECT *
    FROM "s3://bucket/features"
    WHERE features[0] > 0.5
""")

# Aggregate array operations
results = deeplake.query("""
    SELECT AVG(features, axis=0)
    FROM "s3://bucket/features"
""")
```

## Joining Datasets

Join data across different datasets and across different clouds:

```python
# Join datasets from different storage
results = deeplake.query("""
    SELECT i.image, i.embedding, m.labels, m.metadata
    FROM "s3://bucket1/images" AS i
    JOIN "s3://bucket2/metadata" AS m 
    ON i.id = m.image_id
    WHERE m.verified = true
""")

# Complex join with filtering
results = deeplake.query("""
    SELECT 
        i.image,
        e.embedding,
        l.label
    FROM "s3://bucket1/images" AS i
    JOIN "gcs://bucket2/embeddings" AS e ON i.id = e.image_id
    JOIN "azure://container/labels" AS l ON i.id = l.image_id
    WHERE l.confidence > 0.9
    ORDER BY COSINE_SIMILARITY(e.embedding, ARRAY[0.1, 0.2, 0.3]) DESC
    LIMIT 100
""")
```

## Filtering

Filter data using WHERE clauses:

```python
# Simple filters
results = deeplake.query("""
    SELECT *
    FROM "s3://bucket/dataset"
    WHERE label = 'cat'
    AND confidence > 0.9
""")

# Combine with vector search
results = deeplake.query("""
    SELECT *
    FROM "s3://bucket/dataset"
    WHERE label IN ('cat', 'dog')
    ORDER BY COSINE_SIMILARITY(embeddings, ARRAY[0.1, 0.2, 0.3]) DESC
    LIMIT 100
""")
```

## Query Results

Process query results:

```python
# Iterate through results
for item in results:
    image = item["images"]
    label = item["label"]

# Direct column access (recommended for performance)
images = results["images"][:]
labels = results["labels"][:]
```

## Async Queries

Execute queries asynchronously:

```python
# Run query asynchronously
future = deeplake.query_async("""
    SELECT *
    FROM "s3://bucket/dataset"
    ORDER BY COSINE_SIMILARITY(embeddings, ARRAY[0.1,0.2, 0.3]) DESC
""")

# Get results when ready
results = future.result()

# Check completion
if future.is_completed():
    results = future.result()
else:
    print("Query still running")
```

## Querying Views

Chain queries on views:

```python
# Initial query
view = deeplake.query("SELECT * FROM \"s3://bucket/animals\"")

# Query on view
cats = view.query("SELECT * WHERE species = 'cat'")

# Further filter
large_cats = cats.query('SELECT * WHERE weight > 10')
```

## Prepared Queries

Prepare queries for reuse with different parameters:

```python
# Prepare a parameterized query
s_executor = deeplake.prepare_query("""
    SELECT *
    FROM "s3://bucket/dataset"
    WHERE label = 'cat'
    AND confidence > 0.5
""")

m_executor = deeplake.prepare_query("""
    SELECT * FROM "s3://bucket/dataset"
    WHERE label = ? AND confidence > ?
""")

# Execute with different parameters
cats_high = s_executor.run_single()

# Batch execution
results = m_executor.run_batch([
    ["cat", 0.9],
    ["dog", 0.8],
    ["bird", 0.7]
])

# Async execution
future = m_executor.run_single_async(["cat", 0.95])
result = future.result()

# Get the query string
print(f"Query: {m_executor.get_query_string()}")
```

## Query Explanation

Analyze query execution plans:

```python
# Explain query performance
explanation = deeplake.explain_query("""
    SELECT * FROM "s3://bucket/large_dataset"
    WHERE category = 'cat'
    ORDER BY COSINE_SIMILARITY(embeddings, ARRAY[0.1, 0.2, 0.3]) DESC
    LIMIT 1000
""")

# Print explanation
print(explanation)

# Get explanation as dictionary
explain_dict = explanation.to_dict()
print(f"Execution plan: {explain_dict}")

# Use explanation to optimize queries
if "index_used" in explain_dict:
    print("Query will use indexes for optimization")
```
