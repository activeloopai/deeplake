---
seo_title: "Best Practices for Deep Lake Storage and Data Management"
description: "Learn how to tune your Deep Lake usage for optimal performance and cost"
---

<!-- test-context
```python
import numpy as np
import deeplake
from PIL import Image
from io import BytesIO
ds = deeplake.create("tmp://")
ds.add_column("column1", "int32")
ds.add_column("column2", "int32")
value1 = 1
value2 = 2
value3 = 3
value4 = 4
class FileMock:
    def read(self):
        image = Image.new("RGB", (10, 10))
        bytes = BytesIO()
        image.save(bytes, format="JPEG")
        return bytes.getvalue()
f = FileMock()
image_ds = deeplake.create("tmp://")
link_image_ds = deeplake.create("tmp://")
```
-->

# Best Practices for Deep Lake Usage

Deep Lake supports wide range of data types and powerful tools to ingest, load and query your data. This page provides tips for optimizing your usage of Deep Lake for best experience and performance.

## Data Ingestion

### Do commits for version control only

When adding new data to the dataset, Deep Lake automatically flushes the data to the storage. No need to commit each time you add data to the dataset. Commit is only needed when you want to create a new version or checkpoint of the dataset.

### Prefer creating schema before adding data

It is recommended to create the dataset schema before ingestion. Deep Lake supports schema evolution by adding columns when there is already data in the dataset.

However, schema evolution can lead to non-intuitive behavior and performance degradation. It is recommended to create the schema before adding data.

There are cases when the values of specific columns are not known before ingestion. For example if you create embeddings dataset. Usually you have text upfront and you need to create embeddings later (maybe on another machine). In this case distinguish the two steps of dataset creation. First create the dataset with text data, append all texts and then have dedicated step which will create embeddings column and fill it with embeddings.

### Select the right data type for your data

Deep Lake provides a wide range of type to store your data. Some data types can be stored with multiple deeplake types. For example you can store images as `deeplake.types.Array(dimensions=3)` or `deeplake.types.Image()`. For this case, it is recommended to use `deeplake.types.Image()` as it allows to store images in compressed format and stores the data more efficiently.

In general if there is a specific type for your data prefer to use that instead of generic arrays.

- Use `deeplake.types.Image()` for images. This allows efficient image compression and decompression.
- Use `deeplake.types.Video()` for videos. This allows efficient video storage with H264 compression.
- Use `deeplake.types.Mesh()` for 3D meshes. This supports both STL and PLY formats.
- Use `deeplake.types.Text()` for text data. This allows efficient text search and indexing.
- Use `deeplake.types.Embedding()` for embeddings. This allows efficient vector similarity search with query.

### Prefer appending data in batches

There are two ways to append data to the dataset.

1. Append data row by row. In this case you have row as a dictionary. You can combine rows into list of dictionaries and append them to the dataset.

    ```python
    ds.append([{"column1": value1, "column2": value2}, {"column1": value3, "column2": value4}])
    ```

2. Append data in batches. In this case you have single dictionary with list of values for each column.

    ```python
    ds.append({"column1": [value1, value3], "column2": [value2, value4]})
    ```

The second method is more efficient as the Deep Lake columnar format is handling the list for column more efficiently. It can bring to significant performance boost.

### Avoid decompressing images when adding them to the dataset

Deep Lake [Image](../../api/types/#deeplake.types.Image) type supports multiple image formats including JPEG, PNG, and TIFF. When adding images to the dataset, no need to decompress images into `numpy.array` or `PIL.Image` objects. Instead you can pass raw bytes and Deep Lake will automatically decompress the images when reading them.

```python
image_ds.add_column("images", deeplake.types.Image("uint8", "jpeg"))
# f = open("image.jpg", "rb")
image_ds.append({"images": f.read()})
```

This will be much faster and use less memory compared to decompressing the images before adding them to the dataset.

### Use [Link Image](../../api/types/#deeplake.types.Link) for cloud storage images

If the images are in the cloud storage and you want to avoid data duplication you can use [Link Image](../../api/types/#deeplake.types.Link) type. In this case you need to pass the image url.

```python
link_image_ds.add_column("images", deeplake.types.Link(deeplake.types.Image()))
link_image_ds.append({"images": ["s3://bucket/image.jpg"]})
```

### Avoid multiprocessing when ingesting data

Deep Lake is designed to use multiple threads to ingest the data. Most of the cases if will utilize all the cores available on the machine. It will also automatically adjust the worker threads according to the available memory and network bandwidth. Using multiprocessing to ingest the data can lead to performance degradation and memory issues.

To parallelize the ingestion, prefer to use multiple machines and ingest the data in parallel. Deep Lake format guarantees data consistency and atomicity across multiple workers.

### Avoid too much random updates of the data

Small data updates like `ds['column1'][i] = value` can be done in Deep Lake datasets. However if you need to update large number of rows (e.g. more than half of the rows) consider creating a new dataset with the updated data.

## Data Access and Querying

### Prefer opening datasets in read-only mode.

When you are not planning to modify the dataset, prefer opening the dataset with [open_read_only](../../api/dataset/#deeplake.open_read_only) method. This will prevent accidental modifications to the dataset and will improve the performance of the data access.

### Prefer batch access instead of row by row access

If you need to iterate over the dataset or specific column, prefer using batches instead of row by row access. 

```python
# Process all columns with batches. Fast approach.
batch_size = 500
for batch in ds.batches(batch_size):
    print(batch["column1"])

# Process single column with batches. Fast approach.
column = ds["column1"]
for i in range(0, len(column), batch_size):
    print(column[i:i+batch_size])

# Process all columns row by row. Slower than batch access.
for row in ds:
    print(row["column1"])

# Process single column row by row. Slower than batch access.
for i in range(len(column)):
    print(column[i])
```

### Use `query` for complex data filtering and search

Deep Lake supports SQL-like queries to filter and search the data. If you need to filter the data based on multiple columns or complex conditions, prefer using the [deeplake.DatasetView.query](../../api/dataset#deeplake.DatasetView) or [deeplake.query](../../api/query) method, instead of doing manual iteration and filtering.

### Avoid accessing the data of the whole column

The column data can be accessed directly by `ds["column_name"][:]`. For the large datasets this can lead to memory issues. Prefer divide the data into batches and process them separately.

### Consider using async data access

Deep Lake supports async data access and query. If your workflow allows async processing and benefits from that, consider using async data access. Please refer to the [Async Data Loader](../../guide/deep-learning/async-data-loader) guide for the details.


## Storage and Data Management

### Understand the storage differences

Deep Lake supports multiple storage backends, which are differentiated by url schema.

- In memory datasets: `mem://dataset_id`. These datasets are stored in memory and are not persisted. They are useful for temporary data storage and testing.
- Local datasets: `file://path/to/dataset`. These datasets are stored on the local disk. They are useful for local development and testing.
- Cloud datasets:
    - AWS S3: `s3://bucket/dataset`.
    - Azure Blob Storage: `az://container/dataset`.
    - Google Cloud Storage: `gs://bucket/dataset`.

    These datasets are useful for storing large datasets and sharing them across multiple machines.

### Use `mem://` for temporary data and testing

In memory datasets are not persisted and are useful for temporary data storage and testing. They are automatically deleted when the process is terminated.

If you created in memory dataset and you want to persist it you can use [deeplake.copy](../../advanced/sync/#copying-datasets) method to copy the dataset to the local or cloud storage.

### Avoid local storage for large datasets

Local storage shows better latency compared to the cloud dataset. However it is not recommended to use local storage for large dataset as with the scale the performance can be degraded.

If you plan to store large data (>=100GB) cloud storage will be more efficient and reliable.

If you have local dataset and you want to move it to the cloud you can use [deeplake.copy](../../advanced/sync/#copying-datasets) method to copy the dataset to the cloud storage.

### Prefer accessing the cloud storage from the same region

When you are using cloud storage, prefer to access the storage from the same region where the storage is located. This will reduce the latency and improve the performance of the data access.
