---
seo_title: "Deep Lake Schemas | Structured Data Templates"
description: "Deeplake builtin schemas to create dataset."
toc_depth: 2
---
# Schemas

Deep Lake provides pre-built schema templates for common data structures.


## Schema Classes

### Schema

Mutable schema definition for datasets.

::: deeplake.Schema
    options:
        heading_level: 4
        members:
            - __getitem__
            - __len__
            - columns

### SchemaView

Read-only schema definition for datasets.

::: deeplake.SchemaView
    options:
        heading_level: 4
        members:
            - __getitem__
            - __len__
            - columns

## Schema Templates

Schema templates are Python dictionaries that define the structure of the dataset. Each schema template is a dictionary with field names as keys and field types as values.

## Text Embeddings Schema

::: deeplake.schemas.TextEmbeddings
    options:
        heading_level: 3

<!-- test-context
```python
import numpy as np
import deeplake
from deeplake import types
from collections.abc import Mapping

# Mock dataset object with schema support
class MockDataset:
    def __init__(self, path="", schema=None):
        self.path = path
        self._schema = schema or {}

    @property
    def schema(self):
        return MockSchema(self._schema)

class MockSchema(Mapping):
    def __init__(self, schema_dict):
        self._schema = schema_dict if schema_dict else {}

    def __getitem__(self, key):
        if key in self._schema:
            return MockColumn(self._schema[key])
        raise KeyError(f"Column '{key}' not found")

    def __len__(self):
        return len(self._schema)

    def __iter__(self):
        return iter(self._schema)

    def pop(self, key):
        return self._schema.pop(key)

class MockColumn:
    def __init__(self, dtype):
        self.dtype = dtype

# Mock types module
class MockTypes:
    class Text:
        def __init__(self, *args, **kwargs):
            pass

    class Image:
        def __init__(self, *args, **kwargs):
            pass

    class Embedding:
        def __init__(self, size, *args, **kwargs):
            self.size = size

    class UInt64:
        def __init__(self, *args, **kwargs):
            pass

    class Dict:
        def __init__(self, *args, **kwargs):
            pass

# Mock schema templates
class MockSchemaTemplates:
    class TextEmbeddings(dict):
        def __init__(self, embedding_size=768):
            super().__init__()
            self["text"] = MockTypes.Text()
            self["embedding"] = MockTypes.Embedding(embedding_size)

    class COCOImages(dict):
        def __init__(self, embedding_size=768, keypoints=False, objects=False):
            super().__init__()
            self["image"] = MockTypes.Image()
            self["embedding"] = MockTypes.Embedding(embedding_size)
            if keypoints:
                self["keypoints"] = MockTypes.Text()
            if objects:
                self["objects"] = MockTypes.Text()

# Set up mocks
ds = MockDataset("tmp://")

def create(*args, **kwargs):
    schema = kwargs.get('schema', {})
    return MockDataset(args[0] if args else "tmp://", schema)

def open(*args, **kwargs):
    # Return a dataset with some common columns for schema testing
    schema = {
        "images": MockTypes.Image(),
        "labels": MockTypes.Text(),
        "embedding": MockTypes.Embedding(768)
    }
    return MockDataset(args[0] if args else "tmp://", schema)

def open_read_only(*args, **kwargs):
    # Return a dataset with some common columns for schema testing
    schema = {
        "images": MockTypes.Image(),
        "labels": MockTypes.Text(),
        "embedding": MockTypes.Embedding(768)
    }
    return MockDataset(args[0] if args else "tmp://", schema)

def from_coco(*args, **kwargs):
    return MockDataset("tmp://coco")

# Monkey patch deeplake module
deeplake.create = create
deeplake.open = open
deeplake.open_read_only = open_read_only
deeplake.from_coco = from_coco
deeplake.types = MockTypes()
deeplake.schemas = MockSchemaTemplates()
deeplake.Schema = MockSchema
deeplake.SchemaView = MockSchema

# Set up some test variables
images_directory = "path/to/images"
instances_annotation = "path/to/instances.json"
keypoints_annotation = "path/to/keypoints.json"
stuff_annotation = "path/to/stuff.json"
```
-->

```python
# Create dataset with text embeddings schema
ds = deeplake.create("s3://bucket/dataset",
    schema=deeplake.schemas.TextEmbeddings(768))

# Customize before creation
schema = deeplake.schemas.TextEmbeddings(768)
schema["text_embedding"] = schema.pop("embedding")
schema["source"] = deeplake.types.Text()
ds = deeplake.create("s3://bucket/dataset", schema=schema)

# Add field to existing schema
schema = deeplake.schemas.TextEmbeddings(768)
schema["language"] = deeplake.types.Text()
ds = deeplake.create("s3://bucket/dataset", schema=schema)
```

## COCO Images Schema

::: deeplake.schemas.COCOImages
    options:
        heading_level: 3

```python
# Basic COCO dataset
ds = deeplake.create("s3://bucket/dataset",
    schema=deeplake.schemas.COCOImages(768))

# With keypoints and object detection
ds = deeplake.create("s3://bucket/dataset",
    schema=deeplake.schemas.COCOImages(
        embedding_size=768,
        keypoints=True,
        objects=True
    ))

# Customize schema
schema = deeplake.schemas.COCOImages(768)
schema["raw_image"] = schema.pop("image")
schema["camera_id"] = deeplake.types.Text()
ds = deeplake.create("s3://bucket/dataset", schema=schema)
```

## Working with Schema Objects

Access and manipulate dataset schemas:

```python
# Access dataset schema
ds = deeplake.open("s3://bucket/dataset")
schema = ds.schema

# Get column definition
image_col = schema["images"]
print(f"Image column type: {image_col.dtype}")

# Get number of columns
num_columns = len(schema)
print(f"Dataset has {num_columns} columns")

# Read-only schema access
ro_ds = deeplake.open_read_only("s3://bucket/dataset")
ro_schema = ro_ds.schema

# Access column definition (read-only)
label_col = ro_schema["labels"]
print(f"Label column type: {label_col.dtype}")
```

## Custom Schema Template

Create custom schema templates:

```python
# Define custom schema
schema = {
    "id": deeplake.types.UInt64(),
    "image": deeplake.types.Image(),
    "embedding": deeplake.types.Embedding(512),
    "metadata": deeplake.types.Dict()
}

# Create dataset with custom schema
ds = deeplake.create("s3://bucket/dataset", schema=schema)

# Modify schema
schema["timestamp"] = deeplake.types.UInt64()
schema.pop("metadata")
schema["image_embedding"] = schema.pop("embedding")
```

## Creating datasets from predefined data formats

### from_coco

Deep Lake provides a pre-built function to translate COCO format datasets into Deep Lake format.

#### Key Features

- **Multiple Annotation Support**: Handles instances, keypoints, and stuff annotations
- **Flexible Storage**: Works with both cloud and local storage
- **Data Preservation**: 
    - Converts segmentation to binary masks
    - Preserves category hierarchies
    - Maintains COCO metadata
- **Development Features**:
    - Progress tracking during ingestion
    - Configurable tensor and group mappings

#### Basic Usage

The basic flow for COCO ingestion is shown below:

```python
import deeplake

ds = deeplake.from_coco(
    images_directory=images_directory,
    annotation_files={
        "instances": instances_annotation,
        "keypoints": keypoints_annotation,
        "stuff": stuff_annotation,
    },
    dest="al://<your_org_id>/<desired_dataset_name>"
)
```


#### Advanced Configuration

You can customize group names and column mappings using `file_to_group_mapping` and `key_to_column_mapping`:

```python
import deeplake

ds = deeplake.from_coco(
    images_directory=images_directory,
    annotation_files={
        "instances": instances_annotation,
        "keypoints": keypoints_annotation,
    },
    dest="al://<your_org_id>/<desired_dataset_name>",
    file_to_group_mapping={
        "instances": "new_instances",
        "keypoints": "new_keypoints1",
    }
)
```

#### Important Notes

- All segmentation polygons and RLEs are converted to stacked binary masks
- The code assumes all annotation files share the same image IDs
- Supports storage options are
    - Deep Lake cloud storage
    - S3
    - Azure Blob Storage
    - Google Cloud Storage
    - Local File System
- Provides progress tracking through optional tqdm integration
