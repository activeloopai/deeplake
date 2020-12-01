from hub.schema import Tensor, Image, Primitive
from hub.schema.features import flatten
import pytest


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
    primitve_object = Primitive(int)
    assert "'int64'" == primitve_object.__str__()


def test_primitive_repr():
    primitve_object = Primitive(int)
    assert "'int64'" == primitve_object.__repr__()


def test_tensor_init():
    with pytest.raises(ValueError):
        tensor_object = Tensor(shape=2, max_shape=(2, 2))


def test_tensor_str():
    tensor_object = Tensor()
    tensor_object_2 = Tensor(shape=(5000,), dtype="<U20")
    assert tensor_object.__str__() == "Tensor(shape=(None,), dtype='float64')"
    assert tensor_object_2.__str__() == "Tensor(shape=(5000,), dtype='<U20')"


def test_tensor_repr():
    tensor_object = Tensor()
    tensor_object_2 = Tensor(shape=(5000,), dtype="<U20")
    assert tensor_object.__repr__() == "Tensor(shape=(None,), dtype='float64')"
    assert tensor_object_2.__repr__() == "Tensor(shape=(5000,), dtype='<U20')"


if __name__ == "__main__":
    test_tensor_flattening()
    test_primitive_str()
    test_primitive_repr()
    test_tensor_init()
    test_tensor_str()
    test_tensor_repr()
