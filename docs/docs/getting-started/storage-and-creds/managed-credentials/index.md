---
seo_title: "Managed Credentials | Secure Cloud Storage Access"
description: "Set Up Managed Credentials To Enable Secure Access To Cloud Storage Resources In Deep Lake, Essential For Managing Large-Scale AI And Machine Learning Data."
---

# Setting up Deep Lake in Your Cloud

## Connecting Data From Your Cloud Using Deep Lake Managed Credentials

You can use Deep Lake while storing data in your own cloud (AWS, Azure, GCS) without data passing through Activeloop Servers. Most Deep Lake services run on the client, and our browser App or Python API directly read/write data from object storage.

Deep Lake Managed credentials should be set up for granting the client access to cloud storage. Managed Credentials use IAM Policies or Federated Credentials on Activeloop's backend to generate temporary credentials that are used by the client do access the data.

![Authentication_With_Managed_Creds.png](images/Authentication_With_Managed_Creds.png)

## Default Storage

Default storage enables you to map the Deep Lake path `al://org_id/dataset_name` to a cloud path of your choice. Subsequently, all datasets created using the Deep Lake path will be stored at the user-specified, and they can be accessed using API tokens and managed credentials from Deep Lake. By default, the default storage is set as Activeloop Storage, and you may change it using the UI below.

<div style="left: 0; width: 100%; height: 0; position: relative; padding-bottom: 56.25%;"><iframe src="https://www.loom.com/embed/962f130397b344cbbfe9168519f22691" style="top: 0; left: 0; width: 100%; height: 100%; position: absolute; border: 0;" allowfullscreen scrolling="no" allow="encrypted-media;"></iframe></div>

!!! note
    In order to visualize data in the Deep Lake browser application, it is necessary to enable CORS in the bucket containing any source data.

## Connecting Deep Lake Dataset in your Cloud to the Deep Lake to App

If you do not set the Default Storage as your own cloud, Datasets in user's clouds can be connected to the [Deep Lake App](https://app.activeloop.ai) using the Python API below. Once a dataset is connected to Deep Lake, it is assigned a Deep Lake path `al://org_id/dataset_name`, and it can be accessed using API tokens and managed credentials from Deep Lake, without continuously having to specify cloud credentials.

#### Connecting Datasets in the Python API

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

def create(*args, **kwargs):
    return ds

create.__signature__ = get_builtin_signature(deeplake.create)
deeplake.create = create

def connect(*args, **kwargs):
    pass

connect.__signature__ = get_builtin_signature(deeplake.connect)
deeplake.connect = connect

def open_read_only(*args, **kwargs):
    return ds

open_read_only.__signature__ = get_builtin_signature(deeplake.open_read_only)
deeplake.open_read_only = open_read_only

def set_creds_key(*args, **kwargs):
    pass

set_creds_key.__signature__ = get_builtin_signature(deeplake.Dataset.set_creds_key)
deeplake.Dataset.set_creds_key = set_creds_key
link_to_sample = ""

```
-->

```python
# Use deeplake.connect to connect a dataset in your cloud to the Deep Lake App
# Managed Credentials (creds_key) for accessing the data 
# (See Managed Credentials above)
ds = deeplake.create('s3://my_bucket/dataset_name',
creds={'creds_key': 'managed_creds_key'}) # or deeplake.open

# Specify your own path and dataset name for 
# future access to the dataset.
# You can also specify different managed credentials, if desired
deeplake.connect(src='s3://my_bucket/dataset_name',
                 dest='al://org_id/dataset_name',
                 creds_key='managed_creds_key')
ds = deeplake.open_read_only('al://org_id/dataset_name', token='my_token')
```

## Using Manage Credentials with Linked Tensors

Managed credentials can be used for accessing data stored in linked tensors. Simply add the managed credentials to the dataset's creds_keys and assign them to each sample.

```python
ds.add_column('images', deeplake.types.Link(deeplake.types.Image()))
ds.set_creds_key('my_creds_key')
ds.append([{"images": link_to_sample}])
```

## Next Steps

- [Provisioning AWS](aws/provisioning.md)
- [Provisioning Azure](azure/provisioning.md)
- [Provisioning GCP](gcp/provisioning.md)
