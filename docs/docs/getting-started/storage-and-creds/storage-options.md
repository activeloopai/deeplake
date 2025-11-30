---
seo_title: "Storage Options | Use Deep Lake With Preferred Storage Option"
description: "Explore Various Storage Options In Deep Lake, To Store and Search Across Data Efficiently"
---

# Storage Options

How to authenticate using Activeloop storage, AWS S3, and Google Cloud Storage.

## Overview

**Deep Lake datasets can be stored locally, or on several cloud storage providers including Deep Lake Storage, AWS S3,
Microsoft Azure, and Google Cloud Storage.**

Datasets are accessed by choosing the correct prefix for the dataset `path` that is passed to methods such
as [deeplake.open(path)](../../api/dataset.md#deeplake.open),
and [deeplake.create(path)](../../api/dataset.md#deeplake.create).

The path prefixes are:

| Storage Location                     | Path                                             | Notes                                                                                         | 
|--------------------------------------|--------------------------------------------------|-----------------------------------------------------------------------------------------------|
| In Memory                            | mem://dataset_id                                 |                                                                                               |
| Local                                | file://local_path                                |                                                                                               |
| Deep Lake Storage                    | al://org_id/dataset_name                         |                                                                                               |
| AWS S3                               | s3://bucket_name/dataset_name                    | Dataset can be connected to Deep Lake via [Managed Credentials](managed-credentials/index.md) |
| Microsoft Azure (Gen2 DataLake Only) | az://account_name/container_name/dataset_name    | Dataset can be connected to Deep Lake via [Managed Credentials](managed-credentials/index.md) |
| Google Cloud                         | gcs://bucket_name/dataset_name                   | Dataset can be connected to Deep Lake via [Managed Credentials](managed-credentials/index.md) |

!!! tip

    Connecting Deep Lake datasets stored in your own cloud via Deep Lake [Managed Credentials](managed-credentials/index.md) is required for accessing enterprise features, and it significantly simplifies dataset access.

## Authentication for each cloud storage provider

### Activeloop Storage and Managed Datasets

In order to access datasets stored in Deep Lake, or datasets in other clouds that
are [managed by Activeloop](managed-credentials/index.md), users must register and authenticate using
the steps in the link below in [User Authentication](../authentication.md)

### AWS S3

Authentication with AWS S3 has 4 options:

1. Use Deep Lake on a machine in the AWS ecosystem that has access to the relevant S3 bucket
   via [AWS IAM](https://aws.amazon.com/iam/), in which case there is no need to pass credentials in order to access
   datasets in that bucket.

1. Configure AWS through the cli using `aws configure`. This creates a credentials file on your machine that is
   automatically access by Deep Lake during authentication.

1. Save the `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and `AWS_SESSION_TOKEN` (optional) in environmental variables
   of the same name, which are loaded as default credentials if no other credentials are specified.

1. Create a dictionary with the `aws_access_key_id`, `aws_secret_access_key`, and `aws_session_token` (optional), and
   pass it to Deep Lake using:

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
def open(path, creds = None, token = None, org_id = None):
    return deeplake.create("tmp://")
deeplake.open = open
content_of_json_file = {}
```
-->

```python
# Low Level API
ds = deeplake.open("s3://my-bucket/dataset",
    creds = {
    "aws_access_key_id": "my_access_key_id",
    "aws_secret_access_key": "my_aws_secret_access_key",
    "aws_session_token": "my_aws_session_token", # Optional
})
```

#### Custom Storage with S3 API

In order to connect to other object storages supporting S3-like API such as [MinIO](https://github.com/minio/minio),
[StorageGrid](https://www.netapp.com/data-storage/storagegrid/) and others, simply add
endpoint_url the creds dictionary.

```python
ds = deeplake.open('s3://...',
   creds = {
   'aws_access_key_id': "my_access_key_id",
   'aws_secret_access_key': "my_aws_secret_access_key",
   'aws_session_token': "my_aws_session_token", # Optional
   'endpoint_url': 'http://localhost:8888'
   })
```

### Microsoft Azure

Authentication with Microsoft Azure has 4 options:

1. Log in from your machine's CLI using az login.

1. Save the `AZURE_STORAGE_ACCOUNT`, `AZURE_STORAGE_KEY`, or other credentials in environmental variables of the same name,
which are loaded as default credentials if no other credentials are specified.

1. Create a dictionary with the `ACCOUNT_KEY` or `SAS_TOKEN` and pass it to Deep Lake using:

    ```python
    ds = deeplake.open('azure://<account_name>/<container_name>/<dataset_name>',
       creds = {
           'account_key': "my_account_key",
           #OR
           'sas_token': "your_sas_token",
    })
    ```
   
### Google Cloud Storage

Authentication with Google Cloud Storage has 2 options:

1. Create a service account, download the JSON file containing the keys, and then pass that file to the `creds` parameter in `deeplake.open('gcs://.....', creds = 'path_to_keys.json')`. It is also possible to manually pass the information from  the JSON file into the creds parameter using:

    ```python
    ds = deeplake.open('gcs://.....',
         creds = content_of_json_file
    )
    ```

1. Authenticate through the browser using the steps below. This requires that the project credentials are stored on your
machine, which happens after gcloud is initialized and logged in through the CLI. Afterwards, creds can be switched to
creds = 'cache'.

    ```python
       ds = deeplake.open('gcs://.....',
          creds = 'browser' # Switch to 'cache' after doing this once
       )
    ```  
