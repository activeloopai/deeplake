---
seo_title: "Activeloop Deep Lake Docs Dataset"
description: "Access Deep Lake Documentation For Complete Setup, API Reference, Guides On Efficient Multi-Modal AI Search, Dataset Management, Cost-Efficient Training, And Retrieval-Augmented Generation."
toc_depth: 4
---

# Dataset Classes

Deep Lake provides three dataset classes with different access levels:


| Class           | Description                                |
| --------------- | ------------------------------------------ |
| Dataset         | Full read-write access with all operations |
| ReadOnlyDataset | Read-only access to prevent modifications  |
| DatasetView     | Read-only view of query results            |

## Creation Methods

::: deeplake
    options:
        heading_level: 3
        members:
            - create
            - like
            - from_parquet
            - from_csv
            - open
            - open_read_only
            - delete
            - delete_async
            - exists
            - query
            - query_async
            - explain_query
            - prepare_query

## Dataset

The main class providing full read-write access.

::: deeplake.Dataset
    options:
        heading_level: 3
        members:
            - add_column
            - remove_column
            - rename_column
            - append 
            - auto_commit_enabled
            - branch
            - branches
            - tag
            - tags
            - commit
            - commit_async
            - history
            - version
            - created_time
            - current_branch
            - delete
            - description
            - id
            - indexing_mode
            - merge
            - metadata
            - name
            - pull
            - pull_async
            - push
            - push_async
            - query
            - query_async
            - prepare_query
            - explain_query
            - refresh
            - refresh_async
            - schema
            - summary
            - set_creds_key
            - creds_key
            - to_csv
            - batches
            - tensorflow
            - pytorch


Read-only version of Dataset. Cannot modify data but provides access to all data and metadata.

::: deeplake.ReadOnlyDataset
    options:
        heading_level: 3
        members:
            - branches
            - current_branch
            - tags
            - tag
            - description
            - history
            - version
            - created_time
            - id
            - metadata
            - name
            - query
            - query_async
            - refresh
            - refresh_async
            - schema
            - explain_query
            - prepare_query
            - summary
            - to_csv
            - tensorflow
            - pytorch
            - batches


Lightweight view returned by queries. Provides read-only access to query results.

::: deeplake.DatasetView
    options:
        heading_level: 3
        members:
            - batches
            - pytorch
            - query
            - query_async
            - schema
            - summary
            - tag
            - tensorflow
            - to_csv

## Class Comparison

### Dataset
- Full read-write access
- Can create/modify columns
- Can append/update/delete data
- Can commit changes (sync and async)
- Can create version tags and branches
- Can push/pull changes (sync and async)
- Can merge branches
- Auto-commit functionality
- Dataset refresh capabilities
- Full metadata access

<!-- test-context
```python
import numpy as np
import deeplake
import io
from deeplake import types

ds = deeplake.create("tmp://")

def get_builtin_signature(func):
    name = func.__name__
    doc = func.__doc__ or ''
    sig = doc.split('\n')[0].strip()
    return f"{name}{sig}"

def create(*args, **kwargs):
    return deeplake._deeplake.create("tmp://")

create.__signature__ = get_builtin_signature(deeplake.create)
deeplake.create = create

def open(*args, **kwargs):
    return io.BytesIO(b"{}")

def open_(*args, **kwargs):
    return ds

open_.__signature__ = get_builtin_signature(deeplake.open)
deeplake.open = open_

def exists_(path):
    return False

exists_.__signature__ = get_builtin_signature(deeplake.exists)
deeplake.exists = exists_


def delete_(path):
    return False

delete_.__signature__ = get_builtin_signature(deeplake.delete)
deeplake.delete = delete_

def open_read_only(*args, **kwargs):
    return ds

open_read_only.__signature__ = get_builtin_signature(deeplake.open_read_only)
deeplake.open_read_only = open_read_only

image_array = np.random.rand(20, 20, 3)

def tf(*args, **kwargs):
    pass

tf.__signature__ = get_builtin_signature(ds.tensorflow)
deeplake.DatasetView.tensorflow = tf

def from_parquet(*args, **kwargs):
    return ds

from_parquet.__signature__ = get_builtin_signature(deeplake.from_parquet)
deeplake.from_parquet = from_parquet

def from_csv(*args, **kwargs):
    return ds

from_csv.__signature__ = get_builtin_signature(deeplake.from_csv)
deeplake.from_csv = from_csv

def to_csv(*args, **kwargs):
    return "csv_data"

to_csv.__signature__ = get_builtin_signature(deeplake.DatasetView.to_csv)
deeplake.DatasetView.to_csv = to_csv

# Mocks for query functions
class FutureMock:
    def __init__(self, result_value=None):
        self._result = result_value or ds

    def result(self):
        return self._result

    def wait(self):
        pass

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

def query(*args, **kwargs):
    return ds

query.__signature__ = get_builtin_signature(deeplake.query)
deeplake.query = query

def query_async(*args, **kwargs):
    return FutureMock()

query_async.__signature__ = get_builtin_signature(deeplake.query_async)
deeplake.query_async = query_async

def explain_query(*args, **kwargs):
    return ExplainQueryResultMock()

explain_query.__signature__ = get_builtin_signature(deeplake.explain_query)
deeplake.explain_query = explain_query

def prepare_query(*args, **kwargs):
    return ExecutorMock()

prepare_query.__signature__ = get_builtin_signature(deeplake.prepare_query)
deeplake.prepare_query = prepare_query

def delete_async(*args, **kwargs):
    return FutureMock()

delete_async.__signature__ = get_builtin_signature(deeplake.delete_async)
deeplake.delete_async = delete_async
```
-->

```python
ds = deeplake.create("s3://bucket/dataset")
# or
ds = deeplake.open("s3://bucket/dataset")

# Can modify
ds.add_column("images", deeplake.types.Image())
ds.add_column("labels", deeplake.types.ClassLabel("int32"))
ds.add_column("confidence", "float32")
ds["labels"].metadata["class_names"] = ["cat", "dog"]   
ds.append([{"images": image_array, "labels": 0, "confidence": 0.9}])
ds.commit()
```

### ReadOnlyDataset
- Read-only access
- Cannot modify data or schema
- Can view all data and metadata
- Can execute queries (sync and async)
- Can refresh dataset state
- Access to version history and branches
- Full schema and property access
- Returned by `open_read_only()`

```python
ds = deeplake.open_read_only("s3://bucket/dataset")

# Can read
image = ds["images"][0]
metadata = ds.metadata

# Cannot modify
# ds.append([...])  # Would raise error
```

### DatasetView
- Read-only access
- Cannot modify data
- Optimized for query results
- Direct integration with ML frameworks (PyTorch, TensorFlow)
- Batch processing capabilities
- Query chaining support
- Export to CSV functionality
- Schema access
- Returned by `query()` and tag operations

```python
# Get view through query
view = ds.query("SELECT *")

# Access data
image = view["images"][0]

# ML framework integration
torch_dataset = view.pytorch()
tf_dataset = view.tensorflow()
```

## Examples

### Querying Data

```python
# Using Dataset
ds = deeplake.open("s3://bucket/dataset")
results = ds.query("SELECT * WHERE labels = 'cat'")

# Using ReadOnlyDataset
ds = deeplake.open_read_only("s3://bucket/dataset")
results = ds.query("SELECT * WHERE labels = 'cat'")

# Using DatasetView
view = ds.query("SELECT * WHERE labels = 'cat'")
subset = view.query("SELECT * WHERE confidence > 0.9")
```

### Data Access

```python
# Common access patterns work on all types
for row in ds:  # Works for Dataset, ReadOnlyDataset, and DatasetView
    image = row["images"]
    label = row["labels"]

# Column access works on all types
images = ds["images"][:]
labels = ds["labels"][:]
```

### Import/Export Data

```python
# Import from Parquet file
ds = deeplake.from_parquet("data.parquet")
# or from bytes
f = open("data.parquet", "rb")
ds = deeplake.from_parquet(f.read())

# Import from CSV file
ds = deeplake.from_csv("data.csv")
# or from bytes
f = open("data.csv", "rb")
ds = deeplake.from_csv(f.read())

# Export query results to CSV
view = ds.query("SELECT * WHERE labels = 'cat'")
import io
output = io.StringIO()
view.to_csv(output)
csv_data = output.getvalue()
```

### Async Operations

```python
# Async query works on all types
future = ds.query_async("SELECT * WHERE labels = 'cat'")
results = future.result()

# Async data access
future = ds["images"].get_async(slice(0, 1000))
images = future.result()

# # Async dataset operations
# future = ds.commit_async("Updated model predictions")
# future.wait()

# # Async push/pull
# ds.push_async("s3://backup/dataset").wait()
# ds.pull_async("s3://upstream/dataset").wait()
```

### Dataset Management

```python
# Check if dataset exists
if deeplake.exists("s3://bucket/dataset"):
    ds = deeplake.open("s3://bucket/dataset")
else:
    ds = deeplake.create("s3://bucket/dataset")

# Auto-commit functionality
ds.auto_commit_enabled = True  # Enable automatic commits

# Refresh dataset to get latest changes
ds.refresh()

# Delete dataset (irreversible!)
deeplake.delete("s3://old-bucket/dataset")

```

### Advanced Query Operations

```python
# Global query functions
results = deeplake.query("SELECT * FROM 's3://dataset' WHERE confidence > 0.9")

# Async global queries
future = deeplake.query_async("SELECT * FROM 's3://dataset' LIMIT 1000")
results = future.result()

# Explain query execution plan
plan = deeplake.explain_query("SELECT * FROM 's3://dataset' WHERE labels = 'cat'")
print(plan)

# Prepare reusable query executor
executor = deeplake.prepare_query("SELECT * FROM 's3://dataset' WHERE score > ?")
```
