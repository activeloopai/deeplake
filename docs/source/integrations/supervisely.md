# Supervisely

## Dataset to Supervisely Project
Here is an example of a conversion of the dataset into Supervisely format.

```python
from hub import Dataset, schema

# Create dataset
ds = Dataset(
    "./dataset",
    shape=(64,),
    schema={
        "image": schema.Image((512, 512, 3)),
    },
)

# transform into Supervisely project
project = ds.to_supervisely("sample-project")
```

## Supervisely Project to Dataset
In this manner, Hub dataset can be created from a supervisely project:

```python
import hub

out_ds = hub.Dataset.from_supervisely("sample-project")
res_ds = out_ds.store("./dataset")
```
