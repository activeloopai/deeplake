# Dataset


## Create
To create and store dataset you would need to define shape and specify the dataset structure (schema). 

For example, to create a dataset `basic` with 4 samples containing images and labels with shape (512, 512) of dtype 'float' in account `username`:

```python
from hub import Dataset, schema
tag = "username/basic"

ds = Dataset(
    tag,
    shape=(4,),
    schema={
        "image": schema.Tensor((512, 512), dtype="float"),
        "label": schema.Tensor((512, 512), dtype="float"),
    },
)
```

## Upload the Data

To add data to the dataset:

```python
ds["image"][:] = np.ones((4, 512, 512))
ds["label"][:] = np.ones((4, 512, 512))
ds.flush()
```

## Load the data

Load the dataset and access its elements:

```python
ds = Dataset('username/basic')

# Use .numpy() to get the numpy array of the element
print(ds["image"][0].numpy())
print(ds["label", 100:110].numpy())
```


## Convert to Pytorch

```python
ds = ds.to_pytorch()
ds = torch.utils.data.DataLoader(
    ds,
    batch_size=8,
    num_workers=2,
)

# Iterate over the data
for batch in ds:
    print(batch["image"], batch["label"])
```
    
## Convert to Tensorflow  

```python
ds = ds.to_tensorflow().batch(8)

# Iterate over the data
for batch in ds:
    print(batch["image"], batch["label"])
```

## Visualize

Make sure visualization works perfectly at [app.activeloop.ai](https://app.activeloop.ai)

## Delete

You can delete your dataset in [app.activeloop.ai](https://app.activeloop.ai/) in a dataset overview tab.

## Issues

If you spot any trouble or have any question, please open a github issue.


## API

```eval_rst
.. autoclass:: hub.Dataset
   :members:
   :no-undoc-members:
   :private-members:
   :special-members:
```

