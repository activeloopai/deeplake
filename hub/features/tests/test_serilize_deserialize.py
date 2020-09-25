from hub.features.features import Primitive,Tensor,FeatureDict 
from hub.features.serialize import serialize
from hub.features.deserialize import deserialize

def test_serialize_deserialize():
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

    original_result = tuple(t._flatten())
    original_paths = [r.path for r in original_result]
    original_shapes = [r.shape for r in original_result]
    origanal_dtypes = [str(r.dtype) for r in original_result]

    serialize_t=serialize(t)
    deserialize_t=deserialize(serialize_t)

    result = tuple(deserialize_t._flatten())
    paths = [r.path for r in result]
    shapes = [r.shape for r in result]
    dtypes = [str(r.dtype) for r in result]

    assert paths == original_paths
    assert shapes == original_shapes
    assert dtypes == origanal_dtypes


if __name__ == "__main__":
    test_serialize_deserialize()
