---
seo_title: "Activeloop Deep Lake Docs Metadata"
description: "Access Deep Lake Documentation For Metadata."
toc_depth: 2
---
# Metadata

Metadata provides key-value storage for datasets and columns.


## Dataset Metadata

::: deeplake.Metadata
    options:
        heading_level: 4
        members:
            - __getitem__
            - __setitem__
            - __contains__
            - keys

<!-- test-context
```python
import numpy as np
import deeplake
from deeplake import types

def get_builtin_signature(func):
    name = func.__name__
    doc = func.__doc__ or ''
    sig = doc.split('\n')[0].strip()
    return f"{name}{sig}"

ds = deeplake.create("tmp://")
ds.add_column("images", types.Image())
ds.add_column("labels", types.ClassLabel("int32"))

def open_read_only(*args, **kwargs):
    return ds

open_read_only.__signature__ = get_builtin_signature(deeplake.open_read_only)
deeplake.open_read_only = open_read_only

```
-->

```python
# Set dataset metadata
ds.metadata["description"] = "Training dataset"
ds.metadata["version"] = "1.0"
ds.metadata["params"] = {
    "image_size": 224,
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225]
}

# Read dataset metadata
description = ds.metadata["description"]
params = ds.metadata["params"]

# List all metadata keys
for key in ds.metadata.keys():
    print(f"{key}: {ds.metadata[key]}")
```

## Column Metadata

::: deeplake.ReadOnlyMetadata
    options:
        heading_level: 4
        members:
            - __getitem__
            - __contains__
            - keys

```python
# Set column metadata
ds["images"].metadata["mean"] = [0.485, 0.456, 0.406]
ds["images"].metadata["std"] = [0.229, 0.224, 0.225]
ds["labels"].metadata["class_names"] = ["cat", "dog", "bird"]

# Read column metadata
mean = ds["images"].metadata["mean"]
class_names = ds["labels"].metadata["class_names"]

# Check if metadata key exists
if "mean" in ds["images"].metadata:
    print("Mean values are available")

# List all metadata keys for a column
print("Available metadata keys:")
for key in ds["images"].metadata.keys():
    print(f"  {key}: {ds['images'].metadata[key]}")

ds.commit() # Commit the changes to the dataset
```

## Advanced Metadata Operations

```python
# Dataset-level metadata operations
dataset_metadata = ds.metadata

# Check if key exists before accessing
if "training_config" in dataset_metadata:
    config = dataset_metadata["training_config"]
else:
    # Set default configuration
    dataset_metadata["training_config"] = {
        "epochs": 100,
        "batch_size": 32,
        "learning_rate": 0.001
    }

# List all dataset metadata
print("Dataset metadata:")
for key in dataset_metadata.keys():
    print(f"  {key}: {dataset_metadata[key]}")

# Column-level metadata operations
image_metadata = ds["images"].metadata

# Store preprocessing parameters
image_metadata["normalization"] = {
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225]
}
image_metadata["resize_dimensions"] = [224, 224]

# Store data statistics
image_metadata["data_info"] = {
    "total_samples": len(ds),
    "channels": 3,
    "format": "RGB"
}
```

## Read-Only Metadata Access

```python
# Access metadata in read-only datasets
ro_ds = deeplake.open_read_only("s3://bucket/dataset")

# Read dataset metadata (read-only)
if "model_version" in ro_ds.metadata:
    version = ro_ds.metadata["model_version"]
    print(f"Model version: {version}")

# Read column metadata (read-only)
if "class_names" in ro_ds["labels"].metadata:
    classes = ro_ds["labels"].metadata["class_names"]
    print(f"Available classes: {classes}")

# List all available metadata keys
print("Dataset metadata keys:", ro_ds.metadata.keys())
print("Labels metadata keys:", ro_ds["labels"].metadata.keys())
```
