---
toc_depth: 0
seo_title: "Deep Lake Types | Multi-Modal Data Type System"
description: "Definition and Examples of types used in Deeplake Datasets and Columns"
---

# Types

Deep Lake provides a comprehensive type system designed for efficient data storage and retrieval. The type system includes basic numeric types as well as specialized types optimized for common data formats like images, embeddings, and text.


Each type can be specified either using the full type class or a string shorthand:

<!-- test-context
```python
import numpy as np
import deeplake
from deeplake import types
from copy import deepcopy

def get_builtin_signature(func):
    name = func.__name__
    doc = func.__doc__ or ''
    sig = doc.split('\n')[0].strip()
    return f"{name}{sig}"

def add_column_(*args, **kwargs):
    pass
add_column_.__signature__ = get_builtin_signature(deeplake.Dataset.add_column)

def append_(*args, **kwargs):
    pass
append_.__signature__ = get_builtin_signature(deeplake.Dataset.append)

class ColumnMock:
    def __init__(self):
        self.index = None
    def create_index(self, *args, **kwargs):
        self.index = args[0]
    def __getitem__(self, key):
        return self

class DatasetMock:
    def __init__(self):
        self.columns = {}
    def add_column(self, *args, **kwargs):
        add_column_(*args, **kwargs)
    def append(self, *args, **kwargs):
        append_(*args, **kwargs)
    def __getitem__(self, key):
        return ColumnMock()
    def query(self, *args, **kwargs):
        return self

ds = DatasetMock()

ColumnMock.create_index.__signature__ = get_builtin_signature(deeplake.Column.create_index)
DatasetMock.query.__signature__ = get_builtin_signature(deeplake.DatasetView.query)

def create(*args, **kwargs):
    return ds

create.__signature__ = get_builtin_signature(deeplake.create)
deeplake.create = create

def open(*args, **kwargs):
    return ds

open.__signature__ = get_builtin_signature(deeplake.open)
deeplake.open = open

def open_read_only(*args, **kwargs):
    return ds

open_read_only.__signature__ = get_builtin_signature(deeplake.open_read_only)
deeplake.open_read_only = open_read_only

frame1 = np.random.rand(20, 20, 3)
frame2 = frame1
frame3 = frame1
emb1 = np.random.rand(768)
emb2 = emb1
emb3 = emb1
```
-->

```python
# Using type class
ds.add_column("col1", deeplake.types.Float32())

# Using string shorthand
ds.add_column("col2", "float32")
```

#### Types determine:
- How data is stored and compressed
- What operations are available
- How the data can be queried and indexed
- Integration with external libraries and frameworks

## Numeric Types

All basic numeric types:

```python
import deeplake

# Integers
ds.add_column("int8", deeplake.types.Int8())      # -128 to 127
ds.add_column("int16", deeplake.types.Int16())    # -32,768 to 32,767
ds.add_column("int32", deeplake.types.Int32())    # -2^31 to 2^31-1
ds.add_column("int64", deeplake.types.Int64())    # -2^63 to 2^63-1

# Unsigned Integers
ds.add_column("uint8", deeplake.types.UInt8())    # 0 to 255
ds.add_column("uint16", deeplake.types.UInt16())  # 0 to 65,535
ds.add_column("uint32", deeplake.types.UInt32())  # 0 to 2^32-1
ds.add_column("uint64", deeplake.types.UInt64())  # 0 to 2^64-1

# Floating Point
ds.add_column("float16", deeplake.types.Float16())  # Half precision
ds.add_column("float32", deeplake.types.Float32())  # Single precision
ds.add_column("float64", deeplake.types.Float64())  # Double precision

# Boolean
ds.add_column("is_valid", deeplake.types.Bool())     # True/False values
```

## Basic Type Functions

::: deeplake.types.Int8
    options:
        heading_level: 3

::: deeplake.types.Int16
    options:
        heading_level: 3

::: deeplake.types.Int32
    options:
        heading_level: 3

::: deeplake.types.Int64
    options:
        heading_level: 3

::: deeplake.types.UInt8
    options:
        heading_level: 3

::: deeplake.types.UInt16
    options:
        heading_level: 3

::: deeplake.types.UInt32
    options:
        heading_level: 3

::: deeplake.types.UInt64
    options:
        heading_level: 3

::: deeplake.types.Float16
    options:
        heading_level: 3

::: deeplake.types.Float32
    options:
        heading_level: 3

::: deeplake.types.Float64
    options:
        heading_level: 3

::: deeplake.types.Bool
    options:
        heading_level: 3

::: deeplake.types.ClassLabel
    options:
        heading_level: 3

### Numeric Indexing

Numeric columns support indexing for efficient comparison operations:

```python
# Create numeric column with inverted index for range queries
ds.add_column("timestamp", deeplake.types.UInt64())

# Create the index manually
ds["timestamp"].create_index(
    deeplake.types.NumericIndex(deeplake.types.Inverted)
)

# Now you can use efficient comparison operations in queries:
# - Greater than: WHERE timestamp > 1609459200
# - Less than: WHERE timestamp < 1640995200  
# - Between: WHERE timestamp BETWEEN 1609459200 AND 1640995200
# - Value list: WHERE timestamp IN (1609459200, 1640995200)
```

::: deeplake.types.Audio
    options:
        heading_level: 2

```python
# Basic audio storage
ds.add_column("audio", deeplake.types.Audio())

# WAV format
ds.add_column("audio", deeplake.types.Audio(
    sample_compression="wav"
))

# MP3 compression (default)
ds.add_column("audio", deeplake.types.Audio(
    sample_compression="mp3"
))

# With specific dtype
ds.add_column("audio", deeplake.types.Audio(
    dtype="uint8",
    sample_compression="wav"
))

# Audio with Link for external references
ds.add_column("audio_links", deeplake.types.Link(
    deeplake.types.Audio(sample_compression="mp3")
))
```

::: deeplake.types.Image
    options:
        heading_level: 2

```python
# Basic image storage
ds.add_column("images", deeplake.types.Image())

# JPEG compression
ds.add_column("images", deeplake.types.Image(
    sample_compression="jpeg"
))

# With specific dtype
ds.add_column("images", deeplake.types.Image(
    dtype="uint8"  # 8-bit RGB
))
```

::: deeplake.types.Embedding
    options:
        heading_level: 2

```python
# Basic embeddings
ds.add_column("embeddings", deeplake.types.Embedding(768))

# With binary quantization for faster search
ds.add_column("embeddings", deeplake.types.Embedding(
    size=768,
    index_type=deeplake.types.EmbeddingIndex(deeplake.types.ClusteredQuantized)
))

# Custom dtype
ds.add_column("embeddings", deeplake.types.Embedding(
    size=768,
    dtype="float32"
))
```

::: deeplake.types.Text
    options:
        heading_level: 2

```python
# Basic text
ds.add_column("text", deeplake.types.Text())

# Text with BM25 index for semantic search
ds.add_column("text2", deeplake.types.Text(
    index_type=deeplake.types.BM25
))

# Text with inverted index for keyword search
ds.add_column("text3", deeplake.types.Text(
    index_type=deeplake.types.Inverted
))

# Text with exact index for whole text matching
ds.add_column("text4", deeplake.types.Text(
    index_type=deeplake.types.Exact
))
```

::: deeplake.types.Dict
    options:
        heading_level: 2

```python
# Store arbitrary key/value pairs
ds.add_column("metadata", deeplake.types.Dict())

# Add data
ds.append([{
    "metadata": {
        "timestamp": "2024-01-01",
        "source": "camera_1",
        "settings": {"exposure": 1.5}
    }
}])
```

::: deeplake.types.Array
    options:
        heading_level: 2

```python
# Fixed-size array
ds.add_column("features", deeplake.types.Array(
    "float32",
    shape=[512]  # Enforces size
))

# Variable-size array
ds.add_column("sequences", deeplake.types.Array(
    "int32",
    dimensions=1  # Allows any size
))
```

## Numeric Indexes

Deep Lake supports indexing numeric columns for faster lookup operations:

```python
from deeplake.types import NumericIndex, Inverted
# Add numeric column and create an inverted index
ds.add_column("scores", "float32")
ds["scores"].create_index(NumericIndex(Inverted))

# Use with TQL for efficient filtering
results = ds.query("SELECT * WHERE CONTAINS(scores, 0.95)")
```

::: deeplake.types.Bytes
    options:
        heading_level: 2

::: deeplake.types.BinaryMask
    options:
        heading_level: 2

```python
# Basic binary mask
ds.add_column("masks", deeplake.types.BinaryMask())

# With compression
ds.add_column("masks", deeplake.types.BinaryMask(
    sample_compression="lz4"
))
```

::: deeplake.types.SegmentMask
    options:
        heading_level: 2

```python
# Basic segmentation mask
ds.add_column("segmentation", deeplake.types.SegmentMask())

# With compression
ds.add_column("segmentation", deeplake.types.SegmentMask(
    dtype="uint8",
    sample_compression="lz4"
))
```

::: deeplake.types.BoundingBox
    options:
        heading_level: 2

```python
# Basic bounding boxes
ds.add_column("boxes", deeplake.types.BoundingBox())

# With specific format
ds.add_column("boxes", deeplake.types.BoundingBox(
    format="ltwh"  # left, top, width, height
))
```

::: deeplake.types.Point
    options:
        heading_level: 2

::: deeplake.types.Polygon
    options:
        heading_level: 2


::: deeplake.types.Video
    options:
        heading_level: 2

::: deeplake.types.Medical
    options:
        heading_level: 2

::: deeplake.types.Mesh
    options:
        heading_level: 2

::: deeplake.types.Struct
    options:
        heading_level: 2

```python
# Define fixed structure with specific types
ds.add_column("info", deeplake.types.Struct({
    "id": deeplake.types.Int64(),
    "name": "text",
    "score": deeplake.types.Float32()
}))

# Add data
ds.append([{
    "info": {
        "id": 1,
        "name": "sample",
        "score": 0.95
    }
}])
```

::: deeplake.types.Sequence
    options:
        heading_level: 2

```python
# Sequence of images (e.g., video frames)
ds.add_column("frames", deeplake.types.Sequence(
    deeplake.types.Image(sample_compression="jpeg")
))

# Sequence of embeddings
ds.add_column("token_embeddings", deeplake.types.Sequence(
    deeplake.types.Embedding(768)
))

# Add data
ds.append([{
    "frames": [frame1, frame2, frame3],  # List of images
    "token_embeddings": [emb1, emb2, emb3]  # List of embeddings
}])
```

::: deeplake.types.Link
    options:
        heading_level: 2

## Index Types

Deep Lake supports several index types for optimizing queries on different data types.

### IndexType Enum

::: deeplake.types.IndexType
    options:
        heading_level: 3

### Text Index Types

::: deeplake.types.TextIndex
    options:
        heading_level: 3

::: deeplake.types.Inverted
    options:
        heading_level: 3

::: deeplake.types.BM25  
    options:
        heading_level: 3

::: deeplake.types.Exact
    options:
        heading_level: 3

### Numeric Index Types

::: deeplake.types.NumericIndex
    options:
        heading_level: 2

### JSON Index Types

::: deeplake.types.JsonIndex
    options:
        heading_level: 2

### Embedding Index Types

::: deeplake.types.EmbeddingIndexType
    options:
        heading_level: 2

::: deeplake.types.EmbeddingIndex
    options:
        heading_level: 2

::: deeplake.types.EmbeddingsMatrixIndexType
    options:
        heading_level: 2

::: deeplake.types.EmbeddingsMatrixIndex
    options:
        heading_level: 2

### Generic Index Wrapper

::: deeplake.types.Index
    options:
        heading_level: 2

```python
# Create numeric index for efficient range queries
ds.add_column("age", deeplake.types.Int32())
ds["age"].create_index(
    deeplake.types.NumericIndex(deeplake.types.Inverted)
)

# Use in queries with comparison operators
results = ds.query("SELECT * WHERE age > 25")
results = ds.query("SELECT * WHERE age BETWEEN 18 AND 65")
results = ds.query("SELECT * WHERE age IN (25, 30, 35)")
```
