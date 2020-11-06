# Getting Started with Hub



## Access public data. Fast

We’ve talked the talk, now let’s walk through how it works: 

```
pip3 install hub
```

You can access public datasets with a few lines of code.

## Log in to hub

Register a free account at [Activeloop](https://app.activeloop.ai)

Then log in to your account by running the following command:
```
hub login
```

## Create and store dataset

Then create a dataset. It can be stored in your local directory or in your hub account.

```python
import numpy as np

from hub import Dataset
from hub.features import ClassLabel, Image

schema = {
    "image": Image((28, 28)),
    "label": ClassLabel(num_classes=10),
}
ds = Dataset("username/dataset", shape=(1000,), schema=schema)

for i in range(len(ds)):
    ds["image", i] = np.ones((28, 28), dtype="uint8")
    ds["label", i] = 3

print(ds["image", 5].numpy())
print(ds["label", 100:110].numpy())
ds.commit()
```

Hub mimics TFDS data types. Before creating dataset you have to mention the details of its shape and features.
This enables more efficient compression, processing and visualizion of the data.

This code creates dataset in *"username/dataset"* folder with overwrite mode. Dataset has 1000 samples. 
In each sample there is an *image* and a *label*.
After this we can loop over dataset and read/write from it.

## **Why commit?**

Since caching is in place, you need to tell program to push final changes to permanent storage. 

NOTE: This action invalidates dataset.

Alternatively you can use following style.

```python
with hub.open(...) as ds:
    pass
```