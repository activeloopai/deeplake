---
seo_title: "Activeloop Deep Lake Docs Version Control"
description: "Access Deep Lake Documentation For Version Control."
toc_depth: 2
---
# Version Control


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
ds.commit()

def open_read_only(*args, **kwargs):
    return ds

open_read_only.__signature__ = get_builtin_signature(deeplake.open_read_only)
deeplake.open_read_only = open_read_only

```
-->

## Version

::: deeplake.Version
    options:
        heading_level: 4
        members:
            - client_timestamp
            - id
            - message
            - open
            - open_async
            - timestamp

```python
# Get current version
version_id = ds.version

# Access specific version
version = ds.history[version_id]
print(f"Version: {version.id}")
print(f"Message: {version.message}")
print(f"Timestamp: {version.timestamp}")

# Open dataset at specific version
old_ds = version.open()

# Open dataset at specific version asynchronously
old_ds_future = version.open_async()
```

## History

::: deeplake.History
    options:
        heading_level: 4
        members:
            - __getitem__
            - __iter__

```python
# View all versions
for version in ds.history:
    print(f"Version {version.id}: {version.message}")
    print(f"Created: {version.timestamp}")

# Get specific version
version = ds.history[version_id]

# Get version by index
first_version = ds.history[0]
latest_version = ds.history[-1]
```

## Branching

::: deeplake.Branch
    options:
        heading_level: 4
        members:
            - base
            - delete
            - id
            - name
            - open
            - open_async
            - rename
            - timestamp

```python
# Create branch
ds.branch("Branch1")

# Access branch
branch = ds.branches["Branch1"]
print(f"Branch: {branch.name}")
print(f"Created: {branch.timestamp}")
print(f"Base: {branch.base}")

# Open dataset at branch
branch_ds = branch.open()

# Open dataset at branch asynchronously
branch_ds_future = branch.open_async()

# Rename branch
branch.rename("Other Branch")

# Delete branch
branch.delete()
```

::: deeplake.Branches
    options:
        heading_level: 4
        members:
            - __str__
            - __getitem__
            - __len__
            - names

```python
# Create branch
ds.branch("B1")

# List all branches
for name in ds.branches.names():
    br = ds.branches[name]
    print(f"Branch: {br.name} based on {br.base}")

# Check number of branches
num_branches = len(ds.branches)

# Access specific branch
branch = ds.branches["main"]

# Common operations with branches
branch_ds = ds.branches["B1"].open()  # Open branch
branch_future = ds.branches["B1"].open_async()  # Async open

# Error handling
try:
    branch = ds.branches["non_existent"]
except deeplake.BranchNotFoundError:
    print("Branch not found")
```

::: deeplake.BranchView
    options:
        heading_level: 4
        members:
            - base
            - id
            - name
            - open
            - open_async
            - timestamp

```python
# Open read-only dataset
ds = deeplake.open_read_only("s3://bucket/dataset")

# Access branch view
branch_view = ds.branches["B1"]
print(f"Branch: {branch_view.name}")
print(f"Created: {branch_view.timestamp}")

# Open branch view
branch_ds = branch_view.open()

# Open branch view asynchronously
branch_future = branch_view.open_async()
```

::: deeplake.BranchesView
    options:
        heading_level: 4
        members:
            - __getitem__
            - __len__
            - names

```python
# Access read-only branches
branches_view = ds.branches

# List branch names
for name in branches_view.names():
    print(f"Found branch: {name}")

# Get specific branch
branch_view = branches_view["B1"]
```

## Tagging

::: deeplake.Tag
    options:
        heading_level: 4
        members:
            - delete
            - id
            - message
            - name
            - open
            - open_async
            - rename
            - timestamp
            - version

```python
# Create tag
ds.tag("v1.0")

# Access tagged version
tag = ds.tags["v1.0"]
print(f"Tag: {tag.name}")
print(f"Version: {tag.version}")

# Open dataset at tag
tagged_ds = tag.open()
# Open dataset at tag asynchronously
tagged_ds_future = tag.open_async()

# Rename tag
tag.rename("v1.0.0")

# Delete tag
tag.delete()
```

::: deeplake.Tags
    options:
        heading_level: 4
        members:
            - __getitem__
            - __len__
            - names

```python
# Create tag
ds.tag("v1.0")  # Tag current version
specific_version = ds.version
ds.tag("v2.0", version=specific_version)  # Tag specific version

# List all tags
for name in ds.tags.names():
    tag = ds.tags[name]
    print(f"Tag {tag.name} points to version {tag.version}")

# Check number of tags
num_tags = len(ds.tags)

# Access specific tag
tag = ds.tags["v1.0"]

# Common operations with tags
latest_ds = ds.tags["v2.0"].open()  # Open dataset at tag
stable_ds = ds.tags["v1.0"].open_async()  # Async open

# Error handling
try:
    tag = ds.tags["non_existent"]
except deeplake.TagNotFoundError:
    print("Tag not found")
```

::: deeplake.TagView
    options:
        heading_level: 4
        members:
            - id
            - message
            - name
            - open
            - open_async
            - timestamp
            - version

```python
# Open read-only dataset
ds = deeplake.open_read_only("s3://bucket/dataset")

# Access tag view
tag_view = ds.tags["v1.0"]
print(f"Tag: {tag_view.name}")
print(f"Version: {tag_view.version}")

# Open dataset at tag
tagged_ds = tag_view.open()

# Open dataset at tag asynchronously
tagged_future = tag_view.open_async()
```

::: deeplake.TagsView
    options:
        heading_level: 4
        members:
            - __getitem__
            - __len__
            - names

```python
# Access read-only tags
tags_view = ds.tags

# List tag names
for name in tags_view.names():
    print(f"Found tag: {name}")

# Get specific tag
tag_view = tags_view["v1.0"]
```
