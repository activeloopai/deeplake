---
seo_title: "Activeloop Deep Lake Docs | Multi-Modal AI Search | API Reference & Guides"
description: "Access Deep Lake Documentation For Complete Setup, API Reference, Guides On Efficient Multi-Modal AI Search, Dataset Management, Cost-Efficient Training, And Retrieval-Augmented Generation."
hide:
  - toc
---

# 🌊 Deep Lake: Multi-Modal AI Database

Deep Lake is a database specifically designed for machine learning and AI applications, offering efficient data management, vector search capabilities, and seamless integration with popular ML frameworks.

## Key Features

### 🔍 Vector Search & Semantic Operations
- High-performance similarity search for embeddings
- BM25-based semantic text search
- Support for building RAG applications
- Efficient indexing strategies for large-scale search

### 🚀 Optimized for Machine Learning
- Native integration with PyTorch and TensorFlow
- Efficient batch processing for training
- Built-in support for common ML data types (images, embeddings, tensors)
- Automatic data streaming with smart caching

### ☁️ Cloud-Native Architecture
- Native support for major cloud providers:
    - Amazon S3
    - Google Cloud Storage
    - Azure Blob Storage
- Cost-efficient data management
- Data versioning and lineage tracking

## Quick Installation

```bash
pip install deeplake
```

## Basic Usage

<!-- test-context
```python
import numpy as np
import deeplake
from deeplake import types

real_create = deeplake._deeplake.create
def create(path, creds = None, token = None, org_id = None):
    return real_create("tmp://")
deeplake.create = create

image_array = np.random.rand(20, 20, 3)
embedding_vector = np.random.rand(768)
search_vector = np.random.rand(768)
imgs = np.random.rand(3, 20, 20, 3)
bboxes = np.random.rand(3, 20, 4)
smasks = np.random.rand(3, 20, 20, 3)

```
-->

```python
import deeplake

# Create a dataset
ds = deeplake.create("s3://my-bucket/dataset")  # or local path

# Add data columns
ds.add_column("images", deeplake.types.Image())
ds.add_column("embeddings", deeplake.types.Embedding(768))
ds.add_column("labels", deeplake.types.Text())

# Add data
ds.append([{
    "images": image_array,
    "embeddings": embedding_vector,
    "labels": "cat"
}])

# Vector similarity search
text_vector = ','.join(str(x) for x in search_vector)
results = ds.query(f"""
    SELECT *
    ORDER BY COSINE_SIMILARITY(embeddings, ARRAY[{text_vector}]) DESC
    LIMIT 100
""")
```

## Common Use Cases

### Deep Learning Training
```python
# PyTorch integration
from torch.utils.data import DataLoader

loader = DataLoader(ds.pytorch(), batch_size=32, shuffle=True)
for batch in loader:
    images = batch["images"]
    labels = batch["labels"]
    # training code...
```

### RAG Applications
```python
ds = deeplake.create("s3://my-bucket/dataset")  # or local path
# Store text and embeddings
ds.add_column("text", deeplake.types.Text(index_type=deeplake.types.BM25))
ds.add_column("embeddings", deeplake.types.Embedding(1536))

# Semantic search
results = ds.query("""
    SELECT text
    ORDER BY BM25_SIMILARITY(text, 'machine learning') DESC
    LIMIT 10
""")
```

### Computer Vision
```python
# Store images and annotations
ds = deeplake.create("s3://my-bucket/dataset")  # or local path
ds.add_column("images", deeplake.types.Image(sample_compression="jpeg"))
ds.add_column("boxes", deeplake.types.BoundingBox())
ds.add_column("masks", deeplake.types.SegmentMask(sample_compression='lz4'))

# Add data
ds.append({
    "images": imgs,
    "boxes": bboxes,
    "masks": smasks
})
```

## Next Steps

- Check out our [Quickstart Guide](getting-started/quickstart) for detailed setup
- Explore [RAG Applications](guide/rag)
- See [Deep Learning Integration](guide/deep-learning/deep-learning)

## Resources

- [GitHub Repository](https://github.com/activeloopai/deeplake)
- [API Reference](api/)
- [Community Support](https://slack.activeloop.ai)

## Why Deep Lake?

- **Performance**: Optimized for ML workloads with efficient data streaming
- **Scalability**: Handle billions of samples directly from the cloud
- **Flexibility**: Support for all major ML frameworks and cloud providers
- **Cost-Efficiency**: Smart storage management and compression
- **Developer Experience**: Simple, intuitive API with comprehensive features
