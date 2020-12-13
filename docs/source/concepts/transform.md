# Data Pipelines

Data pipelines are usually a series of data transformations on datasets. 

## Transform
Hub Transform provides a functionality to modify the samples of the dataset or create a new dataset from the existing one. 
To apply these modifications user needs to add a `@hub.transform` decorator to any custom generator function. 

## Examples

Basic transform pipeline creation:

```python
my_schema = {
    "image": Tensor((28, 28, 4), "int32", (28, 28, 4)),
    "label": Text(shape=(None,), max_shape=(20,)),
    "confidence": "float",
}

ds = hub.Dataset(
   "./data/test/test_pipeline_basic", mode="w+", shape=(100,), schema=my_schema
)

for i in range(len(ds)):
   ds["image", i] = np.ones((28, 28, 4), dtype="int32")
   ds["label", i] = f"hello {i}"
   ds["confidence", i] = 0.2

@hub.transform(schema=my_schema)
def my_transform(sample, multiplier: int = 2):
   return {
      "image": sample["image"].compute() * multiplier,
      "label": sample["label"].compute(),
      "confidence": sample["confidence"].compute() * multiplier
   }

out_ds = my_transform(ds, multiplier=2)
res_ds = out_ds.store("./data/test/test_pipeline_basic_output")
```

Transormation function can return either a dictionary that corresponds to the provided schema or a list of such dictionaries. In that case the number of samples in the final dataset will be equal to the number of all the returned dictionaries:

```python
dynamic_schema = {
    "image": Tensor(shape=(None, None, None), dtype="int32", max_shape=(32, 32, 3)),
    "label": Text(shape=(None,), max_shape=(20,)),
}

ds = hub.Dataset(
        "./data/test/test_pipeline_dynamic3", mode="w+", shape=(1,), schema=dynamic_schema, cache=False
    )
    
ds["image", 0] = np.ones((30, 32, 3))

@hub.transform(schema=dynamic_schema, scheduler="threaded", nodes=8)
def dynamic_transform(sample, multiplier: int = 2):
   return [{
      "image": sample["image"].compute() * multiplier,
      "label": sample["label"].compute(),
   } for i in range(4)]

out_ds = dynamic_transform(ds, multiplier=4).store("./data/test/test_pipeline_dynamic_output2")
```

You can use transform with multuple processes by adding `scheduler` and `nodes` arguments:

```python

my_schema = {
   "image": Tensor((width, width, channels), dtype, (width, width, channels), chunks=(sample_size // 20, width, width, channels)),
}

@hub.transform(schema=my_schema, scheduler="processed", nodes=2)
def my_transform(x):

   a = np.random.random((width, width, channels))
   for i in range(10):
         a *= np.random.random((width, width, channels))

   return {
         "image": (np.ones((width, width, channels), dtype=dtype) * 255),
   }

ds = hub.Dataset(
   "./data/test/test_pipeline_basic_4", mode="w+", shape=(sample_size,), schema=my_schema, cache=0
)

ds_t = my_transform(ds).store("./data/test/test_pipeline_basic_4")
```

## Ray Transform

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
