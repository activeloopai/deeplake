import numpy as np

import hub.api.dataset as dataset
from hub.features import Tensor

Dataset = dataset.Dataset

my_dtype = {
    "image": Tensor((10, 1920, 1080, 3), "uint8"),
    "label": {
        "a": Tensor((100, 200), "int32"),
        "b": Tensor((100, 400), "int64"),
    },
}


def test_dataset():
    ds = Dataset(
        "./data/hello_world", token=None, num_samples=10000, mode="w+", dtype=my_dtype
    )
    ds["label/a", 5, 50, 50] = 8
    assert ds["label/a", 5, 50, 50] == 8
    ds["image", 5, 4, 100:200, 150:300, :] = np.ones((100, 150, 3), "uint8")
    assert (
        ds["image", 5, 4, 100:200, 150:300, :] == np.ones((100, 150, 3), "uint8")
    ).all()


if __name__ == "__main__":
    test_dataset()