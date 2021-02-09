"""
License:
This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

from typing import Type
from hub.schema import Tensor, Image, Primitive
from hub.schema.features import flatten
import pytest


def test_tensor_error():
    try:
        Tensor(None, max_shape=None)
    except TypeError as ex:
        assert "shape cannot be None" in str(ex)


def test_tensor_error_2():
    with pytest.raises(TypeError):
        t1 = Tensor(shape=(5.1))
    with pytest.raises(TypeError):
        t2 = Tensor(shape=(5.1,))
    with pytest.raises(TypeError):
        t3 = Tensor(shape=(5, 6), max_shape=(7.2, 8))
    with pytest.raises(ValueError):
        t4 = Tensor(shape=(5, 6), max_shape=(7, 8, 9))
    with pytest.raises(TypeError):
        t5 = Tensor(shape=(5, None), max_shape=(5, None))
    with pytest.raises(TypeError):
        t6 = Tensor(shape=(5, 6), max_shape=(7.2, 8))
    with pytest.raises(ValueError):
        t7 = Tensor(max_shape=(10, 15))
    with pytest.raises(TypeError):
        t8 = Tensor(None)
    with pytest.raises(ValueError):
        t9 = Tensor((5, 6, None))
    with pytest.raises(TypeError):
        t10 = Tensor(max_shape="abc")
    with pytest.raises(TypeError):
        t11 = Tensor(max_shape=(7.4, 2))
    with pytest.raises(ValueError):
        t12 = Tensor(max_shape=[])


def test_tensor_flattening():
    t = {
        "image": Image(shape=(300, 400, 3), dtype="uint8"),
        "label": Tensor(
            shape=(5000,),
            dtype="<U20",
        ),
        "gradient": {
            "x": "int32",
            "y": "int32",
        },
    }
    result = tuple(flatten(t))
    paths = [r[1] for r in result]
    dtypes = [r[0] for r in result]

    assert paths == ["/image", "/label", "/gradient/x", "/gradient/y"]
    assert isinstance(dtypes[0], Image)
    assert isinstance(dtypes[1], Tensor)
    assert isinstance(dtypes[2], Primitive)
    assert isinstance(dtypes[3], Primitive)


def test_primitive_str():
    primitve_object = Primitive("int64")
    assert "'int64'" == primitve_object.__str__()


def test_primitive_repr():
    primitve_object = Primitive("int64")
    assert "'int64'" == primitve_object.__repr__()


def test_tensor_init():
    with pytest.raises(ValueError):
        Tensor(shape=2, max_shape=(2, 2))


def test_tensor_str():
    tensor_object_2 = Tensor(shape=(5000,), dtype="<U20")
    assert tensor_object_2.__str__() == "Tensor(shape=(5000,), dtype='<U20')"


def test_tensor_repr():
    tensor_object_2 = Tensor(shape=(5000,), dtype="<U20")
    assert tensor_object_2.__repr__() == "Tensor(shape=(5000,), dtype='<U20')"


if __name__ == "__main__":
    test_tensor_flattening()
    test_primitive_str()
    test_primitive_repr()
    test_tensor_init()
    test_tensor_str()
    test_tensor_repr()
    test_tensor_error_2()
