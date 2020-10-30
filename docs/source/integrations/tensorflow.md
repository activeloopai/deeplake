# Tensorflow

Here is an example to transform the dataset into Tensorflow form.

```python
from hub import dataset

# Load data
ds = dataset.load("mnist/mnist")

# transform into Tensorflow dataset
ds = ds.to_tensorflow().batch(8)

# Iterate over the data
for batch in ds:
    print(batch["data"], batch["labels"])
```
