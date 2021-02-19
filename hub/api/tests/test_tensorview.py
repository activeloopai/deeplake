"""
License:
This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

from hub import Dataset
from hub.api.datasetview import TensorView
from hub.exceptions import NoneValueException
from hub.schema import Tensor, ClassLabel

import numpy as np
import pytest


my_schema = {
    "image": Tensor((None, None, None, None), "uint8", max_shape=(10, 1920, 1080, 4)),
    "label": ClassLabel(num_classes=3),
}
my_schema2 = {
    "image": Tensor((None, None, None, None), "uint8", max_shape=(10, 1920, 1080, 4)),
    "label": ClassLabel(names=["red", "green", "blue"]),
}

ds = Dataset("./data/test/dataset", shape=(100,), mode="w", schema=my_schema)
ds2 = Dataset("./data/test/dataset2", shape=(5,), mode="w", schema=my_schema2)

ds["label", 0] = 1
ds["label", 1] = 2
ds["label", 2] = 0
ds2["label", 0] = 1
ds2["label", 1] = 2
ds2["label", 2] = 0


def test_tensorview_init():
    with pytest.raises(NoneValueException):
        tensorview_object = TensorView(ds, subpath=None)
    with pytest.raises(NoneValueException):
        tensorview_object_2 = TensorView(dataset=None, subpath="image")


def test_tensorview_bug():
    assert ds["image", 1, 2, 3, 4].slice_ == [1, 2, 3, 4]
    assert ds["image", 1][2, 3, 4].slice_ == [1, 2, 3, 4]
    assert ds["image", 1, 2][3, 4].slice_ == [1, 2, 3, 4]
    assert ds["image", 1, 2, 3][4].slice_ == [1, 2, 3, 4]
    assert ds["image", 1:5, 2:7, 3:6, 4:8].slice_ == [
        slice(1, 5),
        slice(2, 7),
        slice(3, 6),
        slice(4, 8),
    ]
    assert ds["image", 1:5][:, 2:7, 3:6, 4:8].slice_ == [
        slice(1, 5),
        slice(2, 7),
        slice(3, 6),
        slice(4, 8),
    ]
    assert ds["image", 1:5, 2:7][:, :, 3:6, 4:8].slice_ == [
        slice(1, 5),
        slice(2, 7),
        slice(3, 6),
        slice(4, 8),
    ]
    assert ds["image", 1:5, 2:7, 3:6][:, :, :, 4:8].slice_ == [
        slice(1, 5),
        slice(2, 7),
        slice(3, 6),
        slice(4, 8),
    ]


def test_tensorview_getitem():
    images_tensorview = ds["image"]
    with pytest.raises(IndexError):
        images_tensorview["7", 0:1920, 0:1080, 0:3].compute()


def test_tensorview_setitem():
    images_tensorview = ds["image"]
    with pytest.raises(ValueError):
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


def test_check_label_name():
    assert ds["label", 0].compute(label_name=True) == "1"
    assert ds["label", 0:3].compute(label_name=True) == ["1", "2", "0"]
    assert ds2["label", 0].compute() == 1
    assert (ds2["label", 0:3].compute() == np.array([1, 2, 0])).all()
    assert ds2["label", 0].compute(label_name=True) == "green"
    assert ds2["label", 1:4].compute(label_name=True) == ["blue", "red", "red"]
    assert ds2["label"].compute(label_name=True) == [
        "green",
        "blue",
        "red",
        "red",
        "red",
    ]
    assert (
        ds2["image", 0].compute(label_name=True).__repr__()
        == "array([], shape=(0, 0, 0, 0), dtype=uint8)"
    )


if __name__ == "__main__":
    test_tensorview_init()
    test_tensorview_getitem()
    test_tensorview_setitem()
    test_check_slice_bound()
    test_tensorview_str()
    test_tensorview_repr()
