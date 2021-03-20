# Getting Started with Hub

### Quickstart

1. Install Hub
    ```
    pip3 install hub
    ```

2. Register and authenticate to upload datasets to [Activeloop](https://app.activeloop.ai/) store
    ```
    activeloop register
    activeloop login
    
    # alternatively, add username and password as arguments (use on platforms like Kaggle)
    activeloop login -u username -p password
    ```
3. Load a dataset

    ```python
    import hub

    ds = hub.Dataset("activeloop/cifar10_train")
    print(ds["label", :10].compute())
    print(ds["id", 1234].compute())
    print(ds["image", 4321].compute())
    ds.copy("./data/examples/cifar10_train")
    ```

4. Create a dataset
    ```python
    import numpy as np

    import hub
    from hub.schema import ClassLabel, Image

    my_schema = {
        "image": Image((28, 28)),
        "label": ClassLabel(num_classes=10),
    }

    url = "./data/examples/quickstart" # write your {username}/{dataset_name} to make it remotely accessible

    ds = hub.Dataset(url, shape=(1000,), schema=my_schema)
    for i in range(len(ds)):
        ds["image", i] = np.ones((28, 28), dtype="uint8")
        ds["label", i] = 3

    print(ds["image", 5].compute())
    print(ds["label", 100:110].compute())
    ds.flush()
    ```
    This code creates dataset in *"./data/examples/new_api_intro"* folder with overwrite mode. Dataset has a thousand samples. In each sample there is an *image* and a *label*. Once the dataset is ready, you may read, write and loop over it.


    You can also transfer a dataset from TFDS (as below) and convert it from/to [Tensorflow](./integrations/tensorflow.md) or [PyTorch](./integrations/pytorch.md).
    ```python
    import hub
    import tensorflow as tf

    out_ds = hub.Dataset.from_tfds('mnist', split='test+train', num=1000)
    res_ds = out_ds.store("username/mnist") # res_ds is now a usable hub dataset
    ```

### Data Storage

Every [dataset](./concepts/dataset.md) needs to specify where it is located. Hub Datasets use its first positional argument to declare its `url`.

#### Hub

If `url` parameter has the form of `username/dataset`, the dataset will be stored in our cloud storage.

```python
url = 'username/dataset'
ds = hub.Dataset(url, shape=(1000,), schema=my_schema)
```

This is the default way to work with Hub datasets. Besides, you can also create or load a dataset locally or in *S3*, *MinIO*, *Google Cloud Storage* and *Azure*.
In case you choose other remote storage platforms, you will need to provide the corresponding credentials as a `token` argument during Dataset creation or loading. It can be a filepath to your credentials or a `dict`.

#### Local storage

To store datasets locally, let the `url` parameter be a local path.
```python
url = './datasets/'
ds = hub.Dataset(url, shape=(1000,), schema=my_schema)
```
#### S3
 ```python
url = 's3://new_dataset'  # your s3 path
ds = hub.Dataset(url, shape=(1000,), schema=my_schema, token={"aws_access_key_id": "...",
                                                              "aws_secret_access_key": "...",
                                                              ...})
```

#### MinIO
```python
url = 's3://new_dataset'  # minio also uses *s3://* prefix
ds = hub.Dataset(url, shape=(1000,), schema=my_schema, token={"aws_access_key_id": "your_minio_access_key",
                                                              "aws_secret_access_key": "your_minio_secret_key",
                                                              "endpoint_url": "your_minio_url:port",
                                                              ...})
```

#### Google Cloud Storage
```python
url = 'gcs://new_dataset' # your google storage (gs://) path
ds = hub.Dataset(url, shape=(1000,), schema=my_schema, token="/path/to/credentials")
```

#### Azure
```python
url = 'https://activeloop.blob.core.windows.net/activeloop-hub/dataset' # Azure link
ds = hub.Dataset(url, shape=(1000,), schema=my_schema, token="/path/to/credentials")
```


### Schema

[Schema](./concepts/features.md) is a dictionary that describes what a dataset consists of. Every dataset is required to have a schema. This is how you can create a simple schema:

```python
from hub.schema import ClassLabel, Image, BBox, Text

my_schema = {
    'kind': ClassLabel(names=["cows", "horses"]),
    'animal': Image(shape=(512, 256, 3)),
    'eyes': BBox(),
    'description': Text(max_shape=(100,))
}
```

### Shape

Shape is another required attribute of a dataset. It simply specifies how large a dataset is. The rules associated with shapes are derived from `numpy`. As you might have noticed, shape is a universal attribute that is also present in schemas, however it is no longer required. If a schema does not have a well-definied shape, `max_shape` might be required.

### Dataset Access, Modification and Deletion

In order to access the data from the dataset, you should use `.compute()` on a portion of the dataset: `ds['key', :5].compute()`.

You can modify the data to the dataset with a regular assignment operator or by performing more sophisticated [transforms](./concepts/transform.md).

You can delete your dataset with `.delete()` or through Activeloop's app on [app.activeloop.ai](https://app.activeloop.ai/) in a dataset overview tab.


### Flush, Commit and Close

Since Hub implements caching, you need to tell the program to push the final changes to permanent storage. Hub Datasets have three methods that let you do that.

The most fundamental method, `.flush()` saves changes from cache to the dataset final storage and does not invalidate dataset object. It means that you can continue working on your data and pushing it later on.

`.commit()` saves the changes into a new version of a dataset that you may go back to later on if you want to.

In rare cases, you may also use `.close()` to invalidate the dataset object after saving the changes.

If you prefer flushing to be taken care for you, wrap your operations on the dataset with the `with` statement in this fashion:
```python
with hub.Dataset(...) as ds:
    pass
```

### Windows FAQ

**Q: Running `activeloop` commands results in an error with a message stating that `'activeloop' is not recognized as an internal or external command, operable program or batch file.` What should I do to use such commands?**

A: If you are having troubles running `activeloop` commands on Windows, it usually means there are issues with your PATH environmental variable and `activeloop` commands are only affected by this underlying problem. Regardless, there are several ways in which you can still be able to use the CLI.

Option 1. You may try running hub as a module, i.e. `py -m hub` and add arguments as necessary.

Option 2. You may try adding Python scripts to your path. First, you need to find out where your Python installation is located. Start from running:
```py --list-paths```
If your Python interpreter is not on the list but you can run it (despite not knowing its path), you should paste the following excerpt to Python console to find out its location:
```python
import os
import sys
os.path.dirname(sys.executable)
```

Once you know the path to the directory with the Python version you are using, adapt it to match the pattern in the command below. If you are unsure whether it is correct, check if the path exists. Finally, run this command in the command prompt (CMD):
<pre>
setx /m PATH "%PATH%;C:\<i>path\to\Python</i>\Python3<i>X</i>\Scripts\"
</pre>

Then refresh your CMD with:
```
start & exit
```
Now, you should be able to run activeloop commands.
