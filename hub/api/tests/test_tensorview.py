from hub import Dataset
from hub.api.tensorview import TensorView
from hub.exceptions import NoneValueException
from hub.schema import Tensor

import numpy as np
import pytest


my_schema = {
    "image": Tensor((None, None, None, None), "uint8", max_shape=(10, 1920, 1080, 4)),
    "label": float,
}

ds = Dataset("./data/test/dataset", shape=(100,), mode="w", schema=my_schema)


def test_tensorview_init():
    with pytest.raises(NoneValueException):
        tensorview_object = TensorView(ds, subpath=None)
    with pytest.raises(NoneValueException):
        tensorview_object_2 = TensorView(dataset=None, subpath="image")


def test_tensorview_getitem():
    images_tensorview = ds["image"]
    with pytest.raises(IndexError):
        images_tensorview["7", 0:1920, 0:1080, 0:3].compute()


def test_tensorview_setitem():
    images_tensorview = ds["image"]
    with pytest.raises(IndexError):
        images_tensorview["7", 0:1920, 0:1080, 0:3] = np.zeros((1920, 1080, 3), "uint8")


def test_check_slice_bound():
    images_tensorview = ds["image"]
    with pytest.raises(ValueError):
        images_tensorview.check_slice_bounds(step=-1)
    with pytest.raises(IndexError):
        images_tensorview.check_slice_bounds(start=1, num=1)
    with pytest.raises(IndexError):
        images_tensorview.check_slice_bounds(start=2, num=1)
    with pytest.raises(IndexError):
        images_tensorview.check_slice_bounds(stop=2, num=1)
    with pytest.raises(IndexError):
        images_tensorview.check_slice_bounds(start=2, stop=1)


def test_tensorview_str():
    images_tensorview = ds["image"]
    assert (
        images_tensorview.__str__()
        == "TensorView(Tensor(shape=(None, None, None, None), dtype='uint8', max_shape=(10, 1920, 1080, 4)), subpath='/image', slice=[slice(0, 100, None)])"
    )


def test_tensorview_repr():
    images_tensorview = ds["image"]
    assert (
        images_tensorview.__repr__()
        == "TensorView(Tensor(shape=(None, None, None, None), dtype='uint8', max_shape=(10, 1920, 1080, 4)), subpath='/image', slice=[slice(0, 100, None)])"
    )


if __name__ == "__main__":
    test_tensorview_init()
    test_tensorview_getitem()
    test_tensorview_setitem()
    test_check_slice_bound()
    test_tensorview_str()
    test_tensorview_repr()
