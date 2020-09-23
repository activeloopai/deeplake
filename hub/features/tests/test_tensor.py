import hub.features as features

Tensor = features.Tensor


def test_tensor_flattening():
    t = Tensor(
        shape=(100, 200),
        dtype={
            "image": Tensor(shape=(300, 400, 3), dtype="uint8"),
            "label": Tensor(
                shape=(5000,),
                dtype={
                    "first": {
                        "a": "<U20",
                        "b": "uint32",
                        "c": Tensor(shape=(13, 17), dtype="float32"),
                    },
                    "second": "float64",
                },
            ),
        },
    )
    result = tuple(t._flatten())
    paths = [r.path for r in result]
    shapes = [r.shape for r in result]
    dtypes = [str(r.dtype) for r in result]

    assert paths == [
        "/image",
        "/label/first/a",
        "/label/first/b",
        "/label/first/c",
        "/label/second",
    ]
    assert shapes == [
        (100, 200, 300, 400, 3),
        (100, 200, 5000),
        (100, 200, 5000),
        (100, 200, 5000, 13, 17),
        (100, 200, 5000),
    ]
    assert dtypes == ["uint8", "<U20", "uint32", "float32", "float64"]


if __name__ == "__main__":
    test_tensor_flattening()