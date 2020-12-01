import numpy as np

import hub
from hub.schema import Tensor


def test_dataset_with_objects():
    schema = {"images": Tensor(shape=(10,), dtype="object", chunks=(5,))}

    ds = hub.Dataset(
        "./data/test/test_dataset_with_objects", mode="w", shape=(100,), schema=schema
    )
    ds["images", 6, 5] = np.ones((20, 30, 4), dtype="uint8")
    ds.close()


if __name__ == "__main__":
    test_dataset_with_objects()
