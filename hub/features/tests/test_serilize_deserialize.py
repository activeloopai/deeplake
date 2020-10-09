from hub.features.features import Tensor
from hub.features.serialize import serialize
from hub.features.deserialize import deserialize
from hub.features.image import Image
from hub.features.class_label import ClassLabel
from hub.features.bbox import BBox
from hub.features.audio import Audio
from hub.features.mask import Mask
from hub.features.polygon import Polygon
from hub.features.segmentation import Segmentation
from hub.features.sequence import Sequence



def test_serialize_deserialize():
    t = Tensor(
        shape=(100, 200),
        dtype={
            "image": Image(shape=(300, 400, 3), dtype="uint8", encoding_format="jpeg"),
            "label": Tensor(
                shape=(5000,),
                dtype={
                    "first": {
                        "a": "<U20",
                        "b": "uint32",
                        "c": ClassLabel(num_classes=3),
                    },
                    "second": "float64",
                },
            ),
            "bbox": BBox(dtype="float64", chunks=False),
            "audio": Audio(shape=(120,), dtype="uint32"),
            "mask": Mask(shape=(5, 8), dtype="uint8", chunks=True),
            "polygon": Polygon(shape=(16, 2)),
            "segmentation": Segmentation(shape=(5, 9, 1), dtype='uint8', names=("apple", "orange", "pineapple")),
            "sequence": Sequence(feature=Tensor(shape=None, dtype="uint8"), length=10)
        },
    )
    original_result = tuple(t._flatten())
    original_paths = [r.path for r in original_result]
    original_shapes = [r.shape for r in original_result]
    origanal_dtypes = [str(r.dtype) for r in original_result]

    serialize_t = serialize(t)
    deserialize_t = deserialize(serialize_t)
    result = tuple(deserialize_t._flatten())
    paths = [r.path for r in result]
    shapes = [r.shape for r in result]
    dtypes = [str(r.dtype) for r in result]

    assert paths == original_paths
    assert shapes == original_shapes
    assert dtypes == origanal_dtypes


if __name__ == "__main__":
    test_serialize_deserialize()
