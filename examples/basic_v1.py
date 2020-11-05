from hub import Dataset, features
import numpy as np

# Tag is set {Username}/{Dataset}
tag = "davitb/basic10"

# Create dataset
ds = Dataset(
    tag,
    shape=(4,),
    schema={
        "image": features.Tensor((512, 512), dtype="float"),
        "label": features.Tensor((512, 512), dtype="float"),
    },
)

# Upload Data
ds["image"][:] = np.zeros((4, 512, 512))
ds["label"][:] = np.zeros((4, 512, 512))
ds.commit()

# Load the data
ds = Dataset(tag)
print(ds["image"][0].compute())