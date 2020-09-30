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
        "./data/test/dataset", token=None, num_samples=10000, mode="w+", dtype=my_dtype
    )
    ds["label/a", 5, 50, 50] = 8
    assert ds["label/a", 5, 50, 50] == 8
    ds["image", 5, 4, 100:200, 150:300, :] = np.ones((100, 150, 3), "uint8")
    assert (
        ds["image", 5, 4, 100:200, 150:300, :] == np.ones((100, 150, 3), "uint8")
    ).all()


my_dtype_with_chunks = {
    "image": Tensor((10, 1920, 1080, 3), "uint8", chunks=(6, 5, 1080, 1080, 3)),
    "label": {
        "a": Tensor((100, 200), "int32", chunks=(6, 100, 200)),
        "b": Tensor((100, 400), "int64", chunks=(6, 50, 200)),
    },
    "another_thing": Tensor(
        (100, 200), Tensor((100, 200), "uint32", chunks=(6, 100, 100, 100, 100))
    ),
}


def test_dataset_with_chunks():
    ds = Dataset(
        "./data/test/dataset_with_chunks",
        token=None,
        num_samples=10000,
        mode="w+",
        dtype=my_dtype_with_chunks,
    )
    ds["label/a", 5, 50, 50] = 8
    assert ds["label/a", 5, 50, 50] == 8
    ds["image", 5, 4, 100:200, 150:300, :] = np.ones((100, 150, 3), "uint8")
    assert (
        ds["image", 5, 4, 100:200, 150:300, :] == np.ones((100, 150, 3), "uint8")
    ).all()


if __name__ == "__main__":
    test_dataset()