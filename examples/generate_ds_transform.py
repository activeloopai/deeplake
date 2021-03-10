import random
import numpy as np

import hub
from hub.schema import Image, ClassLabel


# Create a new dataset
schema = {
    "image": Image(shape=(None, None, None), max_shape=(100, 100, 3), dtype="uint16"),
    "label": ClassLabel(num_classes=3),
    "id": "uint64",
}
tag = "/tmp/ds_classification"
len_ds = 100
ds = hub.Dataset(tag, mode="w+", shape=(len_ds,), schema=schema)


# Transform function
@hub.transform(schema=schema)
def fill_ds(index):
    label = random.choice(range(4))
    if label == 3:
        # Skip the sample
        return []
    return {
        "image": np.ones((100, 100, 3), dtype="int32"),
        "label": label,
        "id": index,
    }


# Fill the dataset and store it
ds = fill_ds(range(100))
ds = ds.store(tag)
print(len(ds))
