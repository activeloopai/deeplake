---
seo_title: "Deep Lake Dataset Synchronization | Copy & Sync Datasets"
description: "Learn how to copy and synchronize Deep Lake datasets across different storage locations and cloud providers"
---

<!-- test-context
```python
import deeplake
import numpy as np
def get_builtin_signature(func):
    name = func.__name__
    doc = func.__doc__ or ''
    sig = doc.split('\n')[0].strip()
    return f"{name}{sig}"

ds = deeplake.create("tmp://")
ds.add_column("images", deeplake.types.Image())
new_images = np.random.rand(10, 10, 10, 3)
batch1 = new_images[:5]
batch2 = new_images[5:]

def create(*args, **kwargs):
    return deeplake._deeplake.create("tmp://")

create.__signature__ = get_builtin_signature(deeplake.create)
deeplake.create = create

def copy(*args, **kwargs):
    pass

copy.__signature__ = get_builtin_signature(deeplake.copy)
deeplake.copy = copy

def open(*args, **kwargs):
    return ds

open.__signature__ = get_builtin_signature(deeplake.open)
deeplake.open = open

old_push = deeplake.Dataset.push
old_pull = deeplake.Dataset.pull

def pull(*args, **kwargs):
    pass

pull.__signature__ = get_builtin_signature(deeplake.Dataset.pull)
deeplake.Dataset.pull = pull

def push(*args, **kwargs):
    pass

push.__signature__ = get_builtin_signature(deeplake.Dataset.push)
deeplake.Dataset.push = push

```
-->

# Dataset Copying and Synchronization

Deep Lake allows copying and synchronizing datasets across different storage locations. This functionality is needed for the following use cases:

- Moving datasets between cloud providers  
- Creating local copies of cloud datasets for faster access
- Backing up datasets to different storage providers
- Maintaining synchronized dataset replicas

## Copying Datasets

Copy a dataset to a new location while preserving all data, metadata, and version history:

```python
# Copy between storage providers
deeplake.copy(
    src="s3://source-bucket/dataset",
    dst="gcs://dest-bucket/dataset", 
    dst_creds={"credentials": "for-dest-storage"}
)

# Create local copy of cloud dataset  
deeplake.copy(
    src="al://org/dataset",
    dst="./local/dataset"
)
```

## Dataset Synchronization 

### Pull Changes

Sync a dataset with its source by pulling new changes:

```python
# Create dataset copy
deeplake.copy("s3://source/dataset", "s3://replica/dataset")

replica_ds = deeplake.open("s3://replica/dataset")

# Later, pull new changes from source
replica_ds.pull(
    url="s3://source/dataset",
    creds={"aws_access_key_id": "key", "aws_secret_access_key": "secret"}
)

# Pull changes asynchronously
async def pull_async():
    await replica_ds.pull_async("s3://source/dataset") 
```

### Push Changes

Push local changes to another dataset location:

```python
# Make changes to dataset
ds.append({"images": new_images})
ds.commit()

# Push changes to replica
ds.push(
    url="s3://replica/dataset",
    creds={"aws_access_key_id": "key", "aws_secret_access_key": "secret"}
)

# Push changes asynchronously  
async def push_async():
    await ds.push_async("s3://replica/dataset")
```

## Synchronization Example

```python
# Initial dataset creation
source_ds = deeplake.create("s3://bucket/source")
source_ds.add_column("images", deeplake.types.Image())
source_ds.commit()

# Create replica
deeplake.copy(
    src="s3://bucket/source",
    dst="gcs://bucket/replica"
)
replica_ds = deeplake.open("gcs://bucket/replica")

# Add data to source
source_ds.append({"images": batch1})
source_ds.commit()

# Sync replica with source
replica_ds.pull("s3://bucket/source")

# Add data to replica  
replica_ds.append({"images": batch2})
replica_ds.commit()

# Push replica changes back to source
replica_ds.push("s3://bucket/source")
```

## Summary

- Copying the dataset preserves all data, metadata, and version history
- Push/pull synchronizes only the changes between datasets
- Copy/sync works across different storage providers - s3, gcs, azure, local, etc.

<!--
```python
deeplake.Dataset.push = old_push
deeplake.Dataset.pull = old_pull
```
-->
