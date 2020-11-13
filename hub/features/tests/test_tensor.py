from hub.features import Tensor, Image, Primitive
from hub.features.features import flatten


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


if __name__ == "__main__":
    test_tensor_flattening()