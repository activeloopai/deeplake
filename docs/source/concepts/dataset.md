# Dataset

## Auto Dataset Creation
If your dataset format is supported, you can point `hub.Dataset` to it's path & allow the `hub.auto` package to infer it's schema & auto convert it into hub format. 

### Supported Dataset Formats
The `hub.auto` package supports the following datasets:

#### Computer Vision
Supports `[.png, .jpg, .jpeg]` file extensions.

- **Image Classification**:
    - Expects the folder path to point to a directory where the folder structure is the following:
        - root
            - class1
                - sample1.jpg
                - sample2.jpg
                - ...
            - class2
                - sample1.png
                - sample2.png
                - ...
            - ...

### Auto Usage
If your dataset is supported (see [above](#supported-dataset-formats)), you can convert it into hub format with a single line of code:

```python
from hub import Dataset

ds = Dataset.from_path("path/to/dataset")
```

### Auto Contribution
If you created & uploaded a dataset into hub, you might as well contribute to the `hub.auto` package. The API for doing so is quite simple:

- If you are writing the ingestion code for a computer vision dataset, then you can create a new file and/or function within `hub.auto.computer_vision`. If your code cannot be organized under preexisting packages/files, you can create new ones & populate the appropriate `__init__.py` files with import code.
- This function should be decorated with `hub.auto.infer.state.directory_parser`. Example:

```python
import hub
from hub.auto.infer import state

# priority is the sort idx of this parser. 
# it's useful for executing more general code first
@state.directory_parser(priority=0)
def image_classification(path, scheduler, workers):
    data_iter = ...

    @hub.transform(schema=schema, scheduler=scheduler, workers=workers)
    def upload_data(sample):
        ...

    # must return a hub dataset, in other words this function should handle
    # reading, transforming, & uploading the dataset into hub format.
    ds = upload_data(data_iter)
    return ds
```

- If you created any new packages/files, make sure to update the [supported dataset formats documentation](#supported-dataset-formats)!


### Best Practice
- Only follow the instructions below for Create/Upload/Load if your dataset is NOT supported by `hub.auto`. 
- This will make your life **significantly easier**.
- If your dataset is not supported, consider [contributing (instructions above)](#auto-contribution)!


## Create
**BEST PRACTICE:** Before you try creating a dataset this way, try following the [Auto Dataset Creation](#auto-dataset-creation) instructions first.

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
**BEST PRACTICE:** Before you try uploading a dataset this way, try following the [Auto Dataset Creation](#auto-dataset-creation) instructions first.

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

