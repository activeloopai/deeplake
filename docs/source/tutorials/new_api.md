## New API

### **Intro**

Today we introduce our new API/format for hub package. It is currently in alpha stage, yet it is very promising.
Eventually we plan to migrate to this format and stick to it for long time. 

Here is some features of new hub: 
1. Ability to modify datasets on fly. Datasets are no longer immutable and can be modified over time
2. Larger datasets can now be uploaded as we removed some RAM limiting components from the hub 
3. Caching is introduced to improve IO performance.
4. Dynamic shaping enables very large images/data support. You can have large images/data stored in hub. 

More features coming: 
 1. Dynamically sized datasets. Soon you will be able to increase number of samples dynamically.
 2. Tensors can be added to dataset on the fly.
 3. Parallelisation to improve IO and data processing.
 4. Better and simplified transformers.
 5. Better dynamic shaping for handling complex metadata.

### **Getting Started**
1) Install alpha version
```
pip3 install hub==1.0.0a4
```

2) Register and authenticate to uploade datasests 
```
hub register
hub login
```

3) Let's start by creating dataset

```python
import numpy as np

import hub
from hub.schema import ClassLabel, Image

schema = {
    "image": Image((28, 28)),
    "label": ClassLabel(num_classes=10),
}

url = "./data/examples/new_api_intro" #instead write your {username}/{dataset} to make it public

ds = hub.Dataset(url, mode="w", shape=(1000,), schema=ds_type)
for i in range(len(ds)):
    ds["image", i] = np.ones((28, 28), dtype="uint8")
    ds["label", i] = 3

print(ds["image", 5].numpy())
print(ds["label", 100:110].numpy())
ds.flush()
```

4) Transferring from TFSDS

In `hub==1.0.0a5` we would also have 
```python
import hub
import tensorflow as tf

out_ds = hub.Dataset.from_tfds('mnist', split='test+train', num=1000)
res_ds = out_ds.store("username/mnist") # res_ds is now a usable hub dataset
```


### Notes 
New hub mimics TFDS data types. Before creating dataset you have to mention the details of what type of data does it contain. This enables us to compress, process and visualize data more efficiently.

This code creates dataset in *"./data/examples/new_api_intro"* folder with overwrite mode. Dataset has 1000 samples. In each sample there is an *image* and a *label*.

After this we can loop over dataset and read/write from it.


### **Why flush?**

Since caching is in place, you need to tell program to push final changes to permanent storage. 

NOTE: This action invalidates dataset.

Alternatively you can use following style.

```python
with hub.Dataset(...) as ds:
    pass
```

This works as well.
