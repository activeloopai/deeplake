---
seo_title: "Deep Lake 3.xx To 4.xx Migration Guide | Optimize Your ML Datasets" 
description: "Step-By-Step Guide For Migrating Deep Lake 3.xx Datasets To 4.xx, Enabling Advanced Vector Search, RAG Applications, And Efficient ML Training."
---

# Migrating to Deep Lake v4

Deep Lake 4.0 introduces major improvements for ML and AI applications:

- Enhanced vector similarity search performance
- Optimized embedding storage and retrieval  
- Improved support for large-scale RAG applications
- Better handling of multi-modal ML data
- Advanced data versioning for ML experiments

This guide walks through migrating your v3 datasets to take advantage of these capabilities.

## Working with v3 Datasets

### Read-Only Access 

While v3 datasets cannot be modified in v4, you can still access them in read-only mode using `deeplake.query()`:

```python 
# Query v3 dataset directly
results = deeplake.query('SELECT * FROM "al://org_name/v3_dataset"')

# Use in ML training pipeline
train_data = results.pytorch()
val_data = results.tensorflow()

# Vector similarity search still works
similar = deeplake.query("""
    SELECT * FROM "al://org_name/v3_dataset"
    ORDER BY COSINE_SIMILARITY(embeddings, ARRAY[...]) DESC 
    LIMIT 100
""")
```

This allows you to continue using existing v3 datasets while gradually migrating to v4.

## Migration Options

### Option 1: Automatic Migration (Recommended)
Use the built-in conversion tool to automatically migrate your dataset:

<!-- test-context
```python
import numpy as np
import deeplake
from deeplake import types

ds = deeplake.create("tmp://")
deeplake.create("mem://old_ds")

def get_builtin_signature(func):
    name = func.__name__
    doc = func.__doc__ or ''
    sig = doc.split('\n')[0].strip()
    return f"{name}{sig}"

def convert(*args, **kwargs):
    pass

convert.__signature__ = get_builtin_signature(deeplake.convert)
deeplake.convert = convert

def create(*args, **kwargs):
    return ds

create.__signature__ = get_builtin_signature(deeplake.create)
deeplake.create = create

search_vector = np.random.rand(768)
old_ds_path = "mem://old_ds"

transforms = None
batch_size = 32
old_ds = ds
new_ds = ds

```
-->

```python
deeplake.convert(
    src='al://org_name/v3_dataset', 
    dst='al://org_name/v4_dataset'
)
```

### Option 2: Manual Migration
For custom schemas or complex ML datasets:

<!-- test-context
```python
ds = deeplake.create("tmp://")
def create(*args, **kwargs):
    return ds

create.__signature__ = get_builtin_signature(deeplake.create)
deeplake.create = create
```
-->

```python
# 1. Create v4 dataset with desired schema
ds = deeplake.create("s3://new/dataset")
ds.add_column("embeddings", deeplake.types.Embedding(768))
ds.add_column("images", deeplake.types.Image()) 
ds.commit()

# 2. Load v3 data through query
source = deeplake.query(f'SELECT * FROM "{old_ds_path}"')

# 3. Migrate in batches with progress tracking
for i in range(0, len(source), batch_size):
    batch = source[i:i+batch_size]
    ds.append(batch)
    if i % 10000 == 0:
        ds.commit()
```

## Validating Your Migration

After migration, verify your ML workflows:

1. Check vector search functionality:
```python
# Verify similarity search
array_str = ','.join(str(x) for x in search_vector)
results = ds.query(f"""
    SELECT * 
    ORDER BY COSINE_SIMILARITY(embeddings, ARRAY[{array_str}]) 
    LIMIT 10
""")
```

2. Validate ML training pipelines:
```python
# Test PyTorch/TensorFlow integration
train_loader = ds.pytorch(transform=transforms)
```

3. Verify data integrity:
```python
# Compare dataset statistics  
assert len(old_ds) == len(new_ds)
```

## Troubleshooting

Common issues during migration:

- **Memory Issues**: For large datasets, use smaller batch sizes during migration
- **Schema Mismatches**: Verify column types match between v3 and v4 datasets  
- **Missing Embeddings**: Ensure embedding dimensions are correctly specified
- **Training Issues**: Update data loading code to use new v4 API

Need help? Join our [Slack Community](https://slack.activeloop.ai)
