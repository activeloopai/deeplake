import numpy as np
from hub import Dataset
from hub.schema import Tensor


def test_tensorview_shapes_1(url="./data/test/dataset", token=None):
    my_schema = {
        "image": Tensor(
            (None, None, None, None), "uint8", max_shape=(10, 1920, 1080, 4)
        ),
        "label": float,
    }
    ds = Dataset(url, token=token, shape=(100,), mode="w", schema=my_schema)
    ds["image", 1] = np.ones((8, 345, 75, 2))
    ds["image", 2] = np.ones((5, 345, 90, 3))
    assert ds["image", 1:3, 2:4, 300:330].shape.tolist() == [
        [2, 30, 75, 2],
        [2, 30, 90, 3],
    ]
    assert ds["image", 0].shape.tolist() == [0, 0, 0, 0]
    assert ds["label", 5:50].shape.tolist() == [45]


def test_tensorview_shapes_2(url="./data/test/dataset", token=None):
    my_schema = {
        "image": Tensor(shape=(None, None, 20), dtype="uint8", max_shape=(10, 50, 20)),
        "fixed": Tensor(shape=(15, 50, 100), dtype="uint8"),
        "label": float,
    }
    ds = Dataset(url, token=token, shape=(100,), mode="w", schema=my_schema)
    ds["image", 4] = np.ones((7, 10, 20))
    assert ds["image", 4, 3:6].shape.tolist() == [3, 10, 20]
    assert ds["image", 4, 3:6, 5:6].shape.tolist() == [3, 1, 20]
    assert ds["fixed", 4, 7:11, 8:9].shape.tolist() == [4, 1, 100]
    assert ds["fixed", 4, 7:11, 8].shape.tolist() == [4, 100]


def test_tensorview_dynamicshapes(url="./data/test/dataset", token=None):
    my_schema = {
        "image": {
            "a": {
                "c": Tensor(
                    shape=(None, None, None),
                    dtype="uint8",
                    max_shape=(10000, 10000, 10000),
                ),
            }
        }
    }
    ds = Dataset(url, token=token, shape=(100,), mode="w", schema=my_schema)
    ds[0:1]["image", "a", "c", 0] = np.ones((7, 10, 20))
    ds[2:3]["image", "a", "c", 0] = 2 * np.ones((7, 10, 20))

    print(ds["image", "a", "c", 2].compute())


if __name__ == "__main__":
    test_tensorview_dynamicshapes()
