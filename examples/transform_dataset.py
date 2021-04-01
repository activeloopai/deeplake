"""Example of using @hub.transform to modify datasets
"""

import os
import numpy as np

import hub

# Load the dataset
ds = hub.Dataset("activeloop/cifar10_train")


# Transform function
@hub.transform(schema=ds.schema, scheduler="threaded")
def add_noise(sample):
    image_shape = sample["image"].shape

    # ADDING NOISE
    noise = np.random.randint(5, size=image_shape, dtype="uint8")

    for i in range(image_shape[0]):
        for j in range(image_shape[1]):
            for k in range(image_shape[2]):
                sample["image"][i][j][k] += noise[i][j][k]
    return sample


ds_transformed = add_noise(ds)

# Compute shards and store
ds_transformed = ds_transformed.store(
    "/tmp/cifar10_train_transformed", sample_per_shard=os.cpu_count()
)
