# Tensorflow

## Dataset to Tensorflow Dataset
Here is an example to transform the dataset into Tensorflow form.

```python
from hub import Dataset

# Create dataset
ds = Dataset(
    "username/tensorflow_example",
    shape=(64,),
    schema={
        "image": schema.Tensor((512, 512), dtype="float"),
        "label": schema.Tensor((512, 512), dtype="float"),
    },
)

# transform into Tensorflow dataset
ds = ds.to_tensorflow().batch(8)

# Iterate over the data
for batch in ds:
    print(batch["image"], batch["label"])
```

## Tensorflow Dataset to Dataset
Hub dataset can be created from tensorflow dataset:

```python
import tensorflow as tf
ds = tf.data.Dataset.from_tensor_slices(tf.range(10))
out_ds = hub.Dataset.from_tensorflow(ds)
res_ds = out_ds.store("./data/from_tf/ds")
```

## TFDS Dataset to Dataset
Also, it is possible to load a dataset using tensorflow_datasets:

```python
import tensorflow_datasets as tfds
with tfds.testing.mock_data(num_examples=5):
    ds = hub.Dataset.from_tfds('mnist', num=5)
    res_ds = ds.store("./data/tfds/mnist", length=5)
```

