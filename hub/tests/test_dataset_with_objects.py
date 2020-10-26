import numpy as np

import hub
from hub.features import Tensor


def test_dataset_with_objects():
    dtype = {"images": Tensor(shape=(10,), dtype="object", chunks=(5,))}

    ds = hub.open(
        "./data/test/test_dataset_with_objects", mode="w", shape=(100,), dtype=dtype
    )
    ds["images", 6, 5] = np.ones((20, 30, 4), dtype="uint8")
    ds.commit()


if __name__ == "__main__":
    test_dataset_with_objects()
