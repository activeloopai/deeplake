import numpy as np

import hub
from hub.features import ClassLabel, Image

ds_type = {
    "image": Image((28, 28)),
    "label": ClassLabel(num_classes=10),
}

ds = hub.open("./data/examples/new_api_intro", mode="w", shape=(1000,), dtype=ds_type)
for i in range(len(ds)):
    ds["image", i] = np.ones((28, 28), dtype="uint8")
    ds["label", i] = 3

print(ds["image", 5].numpy())
print(ds["label", 100:110].numpy())
ds.commit()
