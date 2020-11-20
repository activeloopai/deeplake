import numpy as np
import pytest

from hub import Dataset
from hub.features import Tensor


def test_dynamic_tensor_bug_1(url="./data/test/dataset", token=None):
    my_schema = {
        "image": Tensor(
            (None, 1920, 1080, None), "uint8", max_shape=(10, 1920, 1080, 4)
        ),
    }
    ds = Dataset(url, token=token, shape=(10000,), mode="w", schema=my_schema)
    ds["image", 1] = np.ones((2, 1920, 1080, 1))


def test_dynamic_tensor_bug_2(url="./data/test/dataset", token=None):
    my_schema = {
        "image": Tensor((100, 100), "uint8"),
    }
    ds = Dataset(url, token=token, shape=(10000,), mode="w", schema=my_schema)
    ds["image", 0:1] = [np.zeros((100, 100))]


if __name__ == "__main__":
    # test_dynamic_tensor_bug_1()
    test_dynamic_tensor_bug_2()
