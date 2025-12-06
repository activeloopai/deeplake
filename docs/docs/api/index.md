---
seo_title: "Deep Lake API Reference | Multi-Modal AI Data Management"
description: "Complete API Reference For Deep Lake, Including Dataset Creation, Multi-Modal Query Engine, Vector Search, And ML Framework Integrations."
---

# API Reference

This reference documents the Python API of Deep Lake.

## Core Components

- [Dataset Classes](dataset.md): Documentation for `Dataset`, `ReadOnlyDataset`, and `DatasetView` classes
- [Column Classes](column.md): Documentation for `Column` and `ColumnView` classes
- [Types](types.md): Available data types including basic numeric types and ML-optimized types
- [Schemas](schemas.md): Pre-built schema templates for common data structures
- [Query Language](query.md): Complete TQL syntax and operations
- [Metadata](metadata.md): Dataset and column metadata management
- [Version Control](version_control.md): Version control, history, branches, and tags
- [Miscellaneous](misc.md): Additional auxiliary functionality

## Index Management

Deep Lake supports various index types for optimizing search and query performance:

### Text Indexes
- **BM25**: Full-text search with BM25 similarity scoring
- **Inverted**: Keyword-based text search
- **Exact**: Exact text matching

### Embedding Indexes
- **Clustered**: Default clustering-based embedding search
- **ClusteredQuantized**: Memory-efficient quantized embedding search

### Numeric Indexes
- **Inverted**: Numeric value lookup optimization

Indexes can be created and managed through the `Column` class methods `create_index()` and `drop_index()`. See [Column Classes](column.md) for detailed examples.

## Getting Started

For implementation guidance and examples, please refer to:

- [Quickstart Guide](../getting-started/quickstart.md)
- [Storage Options](../getting-started/storage-and-creds/storage-options.md)
- [Authentication](../getting-started/authentication.md)
