---
seo_title: "Activeloop Deep Lake Docs | Multi-Modal AI Search | API Reference & Guides"
description: "Access Deep Lake Documentation For Complete Setup, API Reference, Guides On Efficient Multi-Modal AI Search, Dataset Management, Cost-Efficient Training, And Retrieval-Augmented Generation."
hide:
  - toc
---

# Quickstart Guide

Get started with Deep Lake by following these examples.

## Installation

Deep Lake can be installed using pip:

```bash
pip install deeplake
```

## Creating a Dataset

<!-- test-context
```python
import numpy as np
import deeplake
from deeplake import types

ds = deeplake.create("tmp://")

real_create = deeplake._deeplake.create
def create(path, creds = None, token = None, org_id = None):
    return real_create("tmp://")
deeplake.create = create

image_array = np.random.rand(20, 20, 3)
video_bytes = b''
embedding_vector = np.random.rand(768)
mask_array = np.random.rand(20, 20, 3)
batch_of_images = np.random.rand(3, 20, 20, 3)
batch_of_videos = [b'', b'', b'']
batch_of_embeddings = np.random.rand(3, 768)
batch_of_masks = np.random.rand(3, 20, 20, 3)

search_vector = np.random.rand(768)

```
-->

```python
import deeplake

# Create a local dataset
ds = deeplake.create("path/to/dataset")

# Or create in cloud storage
ds = deeplake.create("s3://my-bucket/dataset")
ds = deeplake.create("gcs://my-bucket/dataset")
ds = deeplake.create("azure://container/dataset")
```

## Adding Data

Add columns to store different types of data:

```python
# Add basic data types
ds.add_column("ids", "int32")
ds.add_column("labels", "text")

# Add specialized data types
ds.add_column("images", deeplake.types.Image())
ds.add_column("videos", deeplake.types.Video())
ds.add_column("embeddings", deeplake.types.Embedding(768))
ds.add_column("masks", deeplake.types.BinaryMask())
```

Insert data into the dataset:

```python
# Add single samples
ds.append([{
    "ids": 1,
    "labels": "cat",
    "images": image_array,
    "videos": video_bytes,
    "embeddings": embedding_vector,
    "masks": mask_array
}])

# Add batches of data
ds.append({
    "ids": [1, 2, 3],
    "labels": ["cat", "dog", "bird"],
    "images": batch_of_images,
    "videos": batch_of_videos,
    "embeddings": batch_of_embeddings,
    "masks": batch_of_masks
})

ds.commit() # Commit changes to the storage
```

## Accessing Data

Access individual samples:

```python
# Get single items
image = ds["images"][0]
label = ds["labels"][0]
embedding = ds["embeddings"][0]

# Get ranges
images = ds["images"][0:100]
labels = ds["labels"][0:100]

# Get specific indices
selected_images = ds["images"][[0, 2, 3]]
```

## Vector Search

Search by embedding similarity:

```python
# Find similar items
text_vector = ','.join(str(x) for x in search_vector)
results = ds.query(f"""
    SELECT *
    ORDER BY COSINE_SIMILARITY(embeddings, ARRAY[{text_vector}]) DESC
    LIMIT 100
""")

# Process results - Method 1: iterate through items
for item in results:
    image = item["images"]
    label = item["labels"]

# Process results - Method 2: direct column access
images = results["images"][:]
labels = results["labels"][:]  # Recommended for better performance
```

## Data Versioning

```python
# Commit changes
ds.commit("Added initial data")

# Create version tag
ds.tag("v1.0")

# View history
for version in ds.history:
    print(version.id, version.message)

# Create a new branch
ds.branch("new-branch")
### Add new data to the branch ...
main_ds = ds.branches['main'].open()
main_ds.merge("new-branch")
```

## Async Operations

Use async operations for better performance:

```python
# Async data loading
future = ds["images"].get_async(slice(0, 1000))
images = future.result()

# Async query
future = ds.query_async(
    "SELECT * WHERE labels = 'cat'"
)
cats = future.result()
```

## Next Steps

- Explore [RAG applications](../../guide/rag)
- Check out [Deep Learning integration](../../guide/deep-learning/deep-learning)

## Support

If you encounter any issues:

1. Check our [GitHub Issues](https://github.com/activeloopai/deeplake/issues)
2. Join our [Slack Community](https://slack.activeloop.ai)
