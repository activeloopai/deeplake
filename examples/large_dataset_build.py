import numpy as np

import hub
from hub.features import Tensor


def create_large_dataset():
    sample_count = 30  # change this to big number to test
    dtype = {
        "image": Tensor((1920, 1080, 3), chunks=(2, 1920, 1080, 3), dtype="float64")
    }
    array = np.random.random((10, 1920, 1080, 3))
    with hub.open(
        "./data/examples/large_dataset_build",
        mode="w",
        shape=(sample_count,),
        dtype=dtype,
    ) as ds:
        for i in range(len(ds) // 10):
            print(i)
            ds["image", i * 10 : i * 10 + 10] = i * array

    with hub.open("./data/examples/large_dataset_build", mode="r") as ds:
        for i in range(len(ds) // 10):
            assert (ds["image", i * 10, 0, 0, 0].compute() / array[0, 0, 0, 0]) == i


if __name__ == "__main__":
    create_large_dataset()