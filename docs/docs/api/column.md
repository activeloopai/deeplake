---
seo_title: "Deep Lake API Columns"
description: "Access Deep Lake Documentation For Complete Setup, API Reference, Guides On Efficient Multi-Modal AI Search, Dataset Management, Cost-Efficient Training, And Retrieval-Augmented Generation."
toc_depth: 2
---
# Column Classes

Deep Lake provides two column classes for different access levels:


| Class                | Description                                                  |
| -------------------- | ------------------------------------------------------------ |
| Column               | Full read-write access to column data                        |
| ColumnView           | Read-only access to column data                              |
| ColumnDefinition     | Schema definition for columns with modification capabilities |
| ColumnDefinitionView | Read-only schema definition for columns                      |

## Column Class

::: deeplake.Column
    options:
        heading_level: 3
        members:
            - __getitem__
            - __setitem__
            - create_index
            - drop_index
            - dtype
            - get_async
            - get_bytes
            - get_bytes_async
            - indexes
            - metadata
            - name
            - set_async

## ColumnView Class

::: deeplake.ColumnView
    options:
        heading_level: 3
        members:
            - __getitem__
            - dtype
            - get_async
            - get_bytes
            - get_bytes_async
            - indexes
            - metadata
            - name

## ColumnDefinition Class

::: deeplake.ColumnDefinition
    options:
        heading_level: 3
        members:
            - dtype
            - drop
            - name
            - rename

## ColumnDefinitionView Class

::: deeplake.ColumnDefinitionView
    options:
        heading_level: 3
        members:
            - dtype
            - name

## Class Comparison

### Column
- Provides read-write access
- Can modify data and metadata
- Can create/drop indexes for search optimization
- Access to column schema and data type information
- Supports both sync and async operations
- Raw bytes access for binary data
- Available in Dataset

<!-- test-context
```python
import numpy as np
import deeplake
from deeplake import types
from collections.abc import Mapping

# Mock Future class for async operations
class MockFuture:
    def __init__(self, result):
        self._result = result

    def result(self):
        return self._result

    def wait(self):
        pass

# Mock column metadata
class MockMetadata(dict):
    def __init__(self):
        super().__init__()

    def __setitem__(self, key, value):
        super().__setitem__(key, value)

# Mock Column class with full functionality
class MockColumn:
    def __init__(self, name="images", dtype=None):
        self.name = name
        self.dtype = dtype or MockImageType()
        self.metadata = MockMetadata()
        self.indexes = []
        self._data = [np.random.rand(10, 10, 3) for _ in range(1000)]

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._data[key] if key < len(self._data) else np.random.rand(10, 10, 3)
        elif isinstance(key, slice):
            return [self._data[i] if i < len(self._data) else np.random.rand(10, 10, 3) 
                    for i in range(*key.indices(len(self._data)))]
        elif isinstance(key, list):
            return [self._data[i] if i < len(self._data) else np.random.rand(10, 10, 3) 
                    for i in key]
        return np.random.rand(10, 10, 3)

    def __setitem__(self, key, value):
        # Mock assignment
        pass

    def get_async(self, key):
        return MockFuture(self.__getitem__(key))

    def set_async(self, key, value):
        self.__setitem__(key, value)
        return MockFuture(None)

    def get_bytes(self, key):
        # Mock bytes data for image columns
        if hasattr(self.dtype, 'kind') and 'image' in str(self.dtype.kind).lower():
            return b"mock_image_bytes_data"
        return b"mock_bytes_data"

    def get_bytes_async(self, key):
        return MockFuture(self.get_bytes(key))

    def create_index(self, index_type):
        self.indexes.append(str(index_type))

    def drop_index(self, index_type):
        if str(index_type) in self.indexes:
            self.indexes.remove(str(index_type))

# Mock ColumnView (read-only version)
class MockColumnView(MockColumn):
    def __setitem__(self, key, value):
        raise AttributeError("Cannot set item on read-only column")

    def set_async(self, key, value):
        raise AttributeError("Cannot set item on read-only column")

    def create_index(self, index_type):
        raise AttributeError("Cannot create index on read-only column")

    def drop_index(self, index_type):
        raise AttributeError("Cannot drop index on read-only column")

# Mock ColumnDefinition
class MockColumnDefinition:
    def __init__(self, name="images", dtype=None):
        self.name = name
        self.dtype = dtype or MockImageType()

    def rename(self, new_name):
        self.name = new_name

    def drop(self):
        pass

# Mock ColumnDefinitionView (read-only)
class MockColumnDefinitionView:
    def __init__(self, name="images", dtype=None):
        self.name = name
        self.dtype = dtype or MockImageType()

# Mock Schema classes
class MockSchema(Mapping):
    def __init__(self):
        self._columns = {
            "images": MockColumnDefinition("images"),
            "text": MockColumnDefinition("text", MockTextType()),
            "embeddings": MockColumnDefinition("embeddings", MockEmbeddingType())
        }

    def __getitem__(self, key):
        return self._columns.get(key, MockColumnDefinition(key))

    def __len__(self):
        return len(self._columns)

    def __iter__(self):
        return iter(self._columns)

# Mock Dataset classes
class MockDataset:
    def __init__(self, path="", read_only=False):
        self.path = path
        self._read_only = read_only
        self.schema = MockSchema()
        self._columns = {
            "images": MockColumnView("images") if read_only else MockColumn("images"),
            "text": MockColumnView("text", MockTextType()) if read_only else MockColumn("text", MockTextType()),
            "embeddings": MockColumnView("embeddings", MockEmbeddingType()) if read_only else MockColumn("embeddings", MockEmbeddingType())
        }

    def __getitem__(self, key):
        return self._columns.get(key, MockColumnView(key) if self._read_only else MockColumn(key))

    def add_column(self, name, dtype):
        self._columns[name] = MockColumnView(name, dtype) if self._read_only else MockColumn(name, dtype)

    def append(self, data):
        pass

# Mock data types
class MockImageType:
    def __init__(self):
        self.kind = "image"
    def __str__(self):
        return "kind=image, dtype=array(dtype=uint8, shape=(None, None, None))"

class MockTextType:
    def __init__(self):
        self.kind = "text"
    def __str__(self):
        return "kind=text"

class MockEmbeddingType:
    def __init__(self, size=768):
        self.kind = "embedding"
        self.size = size
    def __str__(self):
        return f"kind=embedding, size={self.size}"

# Mock index types
class MockTextIndex:
    def __init__(self, algorithm):
        self.algorithm = algorithm
    def __str__(self):
        return "text_index"

class MockEmbeddingIndex:
    def __init__(self):
        pass
    def __str__(self):
        return "embedding_index"

class MockBM25:
    def __str__(self):
        return "bm25"

# Mock types module
class MockTypes:
    TextIndex = MockTextIndex
    EmbeddingIndex = MockEmbeddingIndex
    BM25 = MockBM25()

    class Image:
        def __init__(self, *args, **kwargs):
            pass

    class Text:
        def __init__(self, *args, **kwargs):
            pass

    class Embedding:
        def __init__(self, size=768, *args, **kwargs):
            self.size = size

# Set up mocks
def create(*args, **kwargs):
    return MockDataset(args[0] if args else "tmp://")

def open(*args, **kwargs):
    return MockDataset(args[0] if args else "s3://bucket/dataset")

def open_read_only(*args, **kwargs):
    return MockDataset(args[0] if args else "s3://bucket/dataset", read_only=True)

# Apply mocks to deeplake
deeplake.create = create
deeplake.open = open
deeplake.open_read_only = open_read_only
deeplake.types = MockTypes()
deeplake.Column = MockColumn
deeplake.ColumnView = MockColumnView
deeplake.ColumnDefinition = MockColumnDefinition
deeplake.ColumnDefinitionView = MockColumnDefinitionView

# Create mock data for examples
ds = create("tmp://")
ds.add_column("images", MockTypes.Image())
ds.append({"images": np.random.rand(1000, 10, 10, 3)})

# Create sample data variables
new_image = np.random.rand(10, 10, 3)
new_batch = np.random.rand(100, 10, 10, 3)

# Get a sample column for the examples 
column = ds["images"]

# Ensure we have a mutable column for index operations
mutable_ds = create("s3://bucket/dataset")
mutable_ds.add_column("text_data", MockTypes.Text())
text_column = mutable_ds["text_data"]

# Override the column variable to ensure it's always mutable for examples
column = MockColumn("images", MockImageType())
```
-->

```python
# Get mutable column
ds = deeplake.open("s3://bucket/dataset")
column = ds["images"]

# Read data
image = column[0]
batch = column[0:100]

# Write data
column[0] = new_image
column[0:100] = new_batch

# Async operations
future = column.set_async(0, new_image)
future.wait()
```

### ColumnView
- Read-only access
- Cannot modify data
- Can read metadata and schema information
- Access to indexes and data type information
- Supports both sync and async operations
- Raw bytes access for binary data
- Available in ReadOnlyDataset and DatasetView

### ColumnDefinition
- Schema-level operations for columns
- Can rename and drop columns
- Access to column data type definitions
- Available through dataset schema

### ColumnDefinitionView
- Read-only schema information
- Access to column data type definitions
- Cannot modify column schema
- Available through read-only dataset schemas

```python
# Get read-only column
ro_ds = deeplake.open_read_only("s3://bucket/dataset")
ro_column = ro_ds["images"]

# Read data
image = ro_column[0]
batch = ro_column[0:100]

# Async read
future = ro_column.get_async(slice(0, 100))
batch = future.result()
```

## Examples

### Data Access

```python
# Direct indexing
single_item = column[0]
batch = column[0:100]
selected = column[[1, 5, 10]]

# Async data access 
future = column.get_async(slice(0, 1000))
data = future.result()
```

### Metadata and Schema Information

```python
# Read metadata from any column type
name = column.name
metadata = column.metadata
data_type = column.dtype

# Update metadata (Column only)
column.metadata["mean"] = [0.485, 0.456, 0.406]
column.metadata["std"] = [0.229, 0.224, 0.225]

# Check column indexes
indexes = column.indexes
print(f"Available indexes: {indexes}")
```

### Index Management

```python
# Create text search index (Column only)
column.create_index(deeplake.types.TextIndex(deeplake.types.BM25))

# Create embedding similarity index (Column only)
column.create_index(deeplake.types.EmbeddingIndex())

# List existing indexes
print(f"Current indexes: {column.indexes}")

# Drop an index
column.drop_index(deeplake.types.TextIndex(deeplake.types.BM25))
```

### Binary Data Access

```python
# Access raw bytes data (useful for images, audio, etc.)
bytes_data = column.get_bytes(0)
batch_bytes = column.get_bytes(slice(0, 10))

# Async bytes access
future = column.get_bytes_async(slice(0, 100))
bytes_batch = future.result()
```

### Schema Operations

```python
# Access column schema information
schema = ds.schema
col_def = schema["images"]
print(f"Column type: {col_def.dtype}")
print(f"Column name: {col_def.name}")

# Modify column schema (Dataset only)
col_def.rename("processed_images")
# col_def.drop()  # Removes entire column
```
