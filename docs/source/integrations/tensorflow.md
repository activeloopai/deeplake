# Tensorflow

Here is an example to transform the dataset into tensorflow form.

```python
from hub import Dataset

# Create dataset
ds = Dataset(
    "username/tensorflow_example",
    shape=(64,),
    schema={
        "image": features.Tensor((512, 512), dtype="float"),
        "label": features.Tensor((512, 512), dtype="float"),
    },
)

# tansform into Tensorflow dataset
ds = ds.to_tensorflow().batch(8)

# Iterate over the data
for batch in ds:
    print(batch["image"], batch["label"])
```
