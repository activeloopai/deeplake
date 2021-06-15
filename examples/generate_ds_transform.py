"""Example of generating hub_v1.Dataset for image classification using @hub_v1.transform

Link to the original dataset: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia
"""
import glob
import os

import numpy as np
import PIL.Image

import hub_v1
from hub_v1.schema import ClassLabel, Image


# Create a new dataset
schema = {
    "image": Image(shape=(None, None, None), max_shape=(3000, 3000, 3), dtype="uint8"),
    "label": ClassLabel(num_classes=2),
}
tag = "/tmp/chest_xray/train"
len_ds = 5216
ds = hub_v1.Dataset(tag, mode="w+", shape=(len_ds,), schema=schema)


# Transform function
@hub_v1.transform(schema=schema, scheduler="threaded", workers=8)
def fill_ds(filename):
    if os.path.basename(os.path.dirname(filename)) == "NORMAL":
        label = 0
    else:
        label = 1
    image = np.array(PIL.Image.open(filename))
    if len(image.shape) == 2:
        image = np.expand_dims(image, -1)
    return {
        "image": image,
        "label": label,
    }


# Fill the dataset and store it
file_list = glob.glob("/home/kristina/Documents/chest_xray/train/*/*.jpeg")
ds = fill_ds(file_list)
ds = ds.store(tag)
print(len(ds))
