# Getting Started with Hub

### Intro

Today we introduce our new API/format for hub package.  

Here is some features of new hub:
1. Ability to modify datasets on fly. Datasets are no longer immutable and can be modified over time.
2. Larger datasets can now be uploaded as we removed some RAM limiting components from the hub.
3. Caching is introduced to improve IO performance.
4. Dynamic shaping enables very large images/data support. You can have large images/data stored in hub. 
5. Dynamically sized datasets. You will be able to increase number of samples dynamically.
6. Tensors can be added to dataset on the fly.

Hub uses [Zarr](https://zarr.readthedocs.io/en/stable/) as a storage for chunked NumPy arrays.

### Getting Started

1. Install beta version
    ```
    pip3 install hub
    ```

2. Register and authenticate to upload datasets
    ```
    hub register
    hub login
    
    # alternatively, add username and password as arguments (use on platforms like Kaggle)
    hub login -u username -p password
    ```

3. Lets start by creating a dataset
```python
import numpy as np

import hub
from hub.schema import ClassLabel, Image

my_schema = {
    "image": Image((28, 28)),
    "label": ClassLabel(num_classes=10),
}

url = "./data/examples/new_api_intro" #instead write your {username}/{dataset} to make it public

ds = hub.Dataset(url, shape=(1000,), schema=my_schema)
for i in range(len(ds)):
    ds["image", i] = np.ones((28, 28), dtype="uint8")
    ds["label", i] = 3

print(ds["image", 5].compute())
print(ds["label", 100:110].compute())
ds.close()
```

You can also transfer a dataset from TFDS.
```python
import hub
import tensorflow as tf

out_ds = hub.Dataset.from_tfds('mnist', split='test+train', num=1000)
res_ds = out_ds.store("username/mnist") # res_ds is now a usable hub dataset
```

### Data Storage

#### Hub

If `url` parameter has the form of `username/dataset`, the dataset will be stored in our cloud storage.

```python
url = 'username/dataset'
```

Besides, you can also create a dataset in *S3*, *MinIO*, *Google Cloud Storage* or *Azure*.
In that case you will need to have the corresponding credentials and provide them as a `token` argument during Dataset creation. It can be a filepath to your credentials or a `dict`.

#### S3
 ```python
url = 's3://new_dataset'  # s3
ds = hub.Dataset(url, shape=(1000,), schema=my_schema, token={"aws_access_key_id": "...",
                                                              "aws_secret_access_key": "...",
                                                              ...})
```

#### MinIO
```python
url = 's3://new_dataset'  # minio also uses `s3://` prefix
ds = hub.Dataset(url, shape=(1000,), schema=my_schema, token={"aws_access_key_id": "your_minio_access_key",
                                                              "aws_secret_access_key": "your_minio_secret_key",
                                                              "endpoint_url": "your_minio_url:port",
                                                              ...})
```

#### Google Cloud Storage
```python
url = 'gcs://new_dataset' # gcloud
ds = hub.Dataset(url, shape=(1000,), schema=my_schema, token="/path/to/credentials")
```

#### Azure
```python
url = 'https://activeloop.blob.core.windows.net/activeloop-hub/dataset' # Azure
ds = hub.Dataset(url, shape=(1000,), schema=my_schema, token="/path/to/credentials")
```
### Deletion

You can delete your dataset in [app.activeloop.ai](https://app.activeloop.ai/) in a dataset overview tab.

### Notes

New hub mimics TFDS data types. Before creating dataset you have to mention the details of what type of data does it contain. This enables us to compress, process and visualize data more efficiently.

This code creates dataset in *"./data/examples/new_api_intro"* folder with overwrite mode. Dataset has 1000 samples. In each sample there is an *image* and a *label*.

After this we can loop over dataset and read/write from it.


### Why flush?

Since caching is in place, you need to tell the program to push final changes to a permanent storage. 

`.close()` saves changes from cache to dataset final storage and does not invalidate dataset object.
On the other hand, `.flush()` saves changes to dataset, but invalidates it.


Alternatively you can use the following style.

```python
with hub.Dataset(...) as ds:
    pass
```

This works as well.
