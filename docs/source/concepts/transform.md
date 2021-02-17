# Data Pipelines

Data pipelines are a series of data transformations on datasets. 

## Transform
Hub Transform provides a functionality to modify the samples of the dataset or create a new dataset from the existing one. 
To apply these modifications user needs to add a `@hub.transform` decorator to any custom function. User defined transform function is applied to every sample on the input. It takes in an iterator or a dataset, and output another dataset with specified schema.

There are optimizations done behind the scenes to efficiently process and store the data. 

### How to upload a dataset with @hub.Transform

Define the desired schema
```python
import hub
from hub.schema import Tensor, Text
import numpy as np

my_schema = {
    "image": Tensor((28, 28, 4), "int32", (28, 28, 4)),
    "label": Text(shape=(None,), max_shape=(20,)),
    "confidence": "float",
}
tag = "./data/test/test_pipeline_basic"
```

Before

```python 
ds = hub.Dataset(
   tag, mode="w+", shape=(100,), schema=my_schema
)

for i in range(len(ds)):
   ds["image", i] = np.ones((28, 28, 4), dtype="int32")
   ds["label", i] = f"hello {i}"
   ds["confidence", i] = 0.2
   
assert ds["confidence"][0].compute() == 0.2
```

After

```python
@hub.transform(schema=my_schema)
def my_transform(index):
    return {
        "image": np.ones((28, 28, 4), dtype="int32"),
        "label": f"hello {index}",
        "confidence": 0.2,
    }


ds = my_transform(range(100))
ds = ds.store(tag)

assert ds["confidence"][0].compute() == 0.2

# Access the dataset later
ds = hub.Dataset(tag)
assert ds["confidence"][0].compute() == 0.2
```

### Adding arguments

```python
@hub.transform(schema=my_schema)
def my_transform(index, multiplier):
    return {
        "image": np.ones((28, 28, 4), dtype="int32"),
        "label": f"hello {index}",
        "confidence": 0.2 * multiplier,
    }


ds = my_transform(range(100), multiplier=10)
ds = ds.store(tag)

assert ds["confidence"][0].compute() == 2.0
```

### Stacking multiple transforms 

```python
@hub.transform(schema=my_schema)
def my_transform_1(index):
    return {
        "image": np.ones((28, 28, 4), dtype="int32"),
        "label": f"hello {index}",
        "confidence": 0.2,
    }


@hub.transform(schema=my_schema)
def my_transform_2(sample, multiplier: int = 2):
    return {
        "image": sample["image"].compute() * multiplier,
        "label": sample["label"].compute(),
        "confidence": sample["confidence"].compute() * multiplier,
    }


ds = my_transform_1(range(100))
ds = my_transform_2(ds, multiplier=10)
ds = ds.store(tag)

assert ds["confidence"][0].compute() == 2.0
```

### Returning multiple elements

Transformation function can return either a dictionary that corresponds to the provided schema or a list of such dictionaries. In that case the number of samples in the final dataset will be equal to the number of all the returned dictionaries:

```python
my_schema = {
    "image": Tensor(shape=(None, None, None), dtype="int32", max_shape=(32, 32, 3)),
    "label": Text(shape=(None,), max_shape=(20,)),
}

ds = hub.Dataset(
    "./data/test/test_pipeline_dynamic3",
    mode="w+",
    shape=(1,),
    schema=my_schema,
    cache=False,
)

ds["image", 0] = np.ones((30, 32, 3))


@hub.transform(schema=my_schema)
def dynamic_transform(sample, multiplier: int = 2):
    return [
        {
            "image": sample["image"].compute() * multiplier,
            "label": sample["label"].compute(),
        }
        for i in range(multiplier)
    ]


out_ds = dynamic_transform(ds, multiplier=4).store("./data/pipeline")
assert len(ds) == 1
assert len(out_ds) == 4
```

### Local parallel execution

You can use transform with multuple processes or threads by setting `scheduler` to `threaded` or `processed` and set number of `workers`.

```python

width = 256
channels = 3
dtype = "uint8"

my_schema = {"image": Image(shape=(width, width, channels), dtype=dtype)}


@hub.transform(schema=my_schema, scheduler="processed", workers=2)
def my_transform(x):

    a = np.random.random((width, width, channels))
    for i in range(10):
        a *= np.random.random((width, width, channels))

    return {
        "image": (np.ones((width, width, channels), dtype=dtype) * 255),
    }


ds_t = my_transform(range(100)).store("./data/pipeline")
```

### Scaling compute to a cluster

[in development]

There is also an option of using `ray` as a scheduler. In this case `RayTransform` will be applied to samples. 

```python
ds = hub.Dataset(
   "./data/ray/ray_pipeline_basic",
   mode="w+",
   shape=(100,),
   schema=my_schema,
   cache=False,
)

for i in range(len(ds)):
   ds["image", i] = np.ones((28, 28, 4), dtype="int32")
   ds["label", i] = f"hello {i}"
   ds["confidence/confidence", i] = 0.2

@hub.transform(schema=my_schema, scheduler="ray")
def my_transform(sample, multiplier: int = 2):
   return {
      "image": sample["image"].compute() * multiplier,
      "label": sample["label"].compute(),
      "confidence": {
            "confidence": sample["confidence/confidence"].compute() * multiplier
      },
   }

out_ds = my_transform(ds, multiplier=2)
```
## API
```eval_rst
.. automodule:: hub.compute
   :members:
   :no-undoc-members:
   :private-members:
   :special-members:
```
```eval_rst
.. autoclass:: hub.compute.transform.Transform
   :members:
   :no-undoc-members:
   :private-members:
   :special-members:
```
```eval_rst
.. autoclass:: hub.compute.ray.RayTransform
   :members:
   :no-undoc-members:
   :private-members:
   :special-members:
```
