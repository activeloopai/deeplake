import numpy as np
import hub
from hub.schema import Tensor


def create_large_dataset():
    sample_count = 60  # change this to big number to test

    # Decide schema of the dataset
    schema = {"image": Tensor((1920, 1080, 3), dtype="float64")}
    array = np.random.random((10, 1920, 1080, 3))

    # Write the dataset
    ds = hub.Dataset(
        "./data/examples/large_dataset_build",
        shape=(sample_count,),
        schema=schema,
    )

    for i in range(len(ds) // 10):
        ds["image", i * 10 : i * 10 + 10] = i * array
    ds.flush()

    ds = hub.Dataset("./data/examples/large_dataset_build")
    print(ds.keys, ds["image"].shape, ds["image"].dtype)

    # Read the dataset
    with hub.Dataset("./data/examples/large_dataset_build") as ds:
        for i in range(len(ds) // 10):
            assert (ds["image", i * 10, 0, 0, 0].compute() / array[0, 0, 0, 0]) == i


if __name__ == "__main__":
    create_large_dataset()
