"""
License:
This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

from hub.schema.video import Video
from hub.schema.text import Text
from hub.schema.sequence import Sequence
import pytest
from hub.schema.features import Tensor, Primitive
from hub.schema.serialize import serialize
from hub.schema.deserialize import deserialize
from hub.schema.image import Image
from hub.schema.class_label import ClassLabel
from hub.schema.bbox import BBox
from hub.schema.audio import Audio
from hub.schema.mask import Mask
from hub.schema.polygon import Polygon
from hub.schema.segmentation import Segmentation


def test_serialize_deserialize():
    t = Tensor(
        shape=(100, 200),
        dtype={
            "image": Image(shape=(300, 400, 3), dtype="uint8"),
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
            "bbox": BBox(dtype="float64"),
            "audio": Audio(shape=(120,), dtype="uint32"),
            "mask": Mask(shape=(5, 8, 1)),
            "polygon": Polygon(shape=(16, 2)),
            "segmentation1": Segmentation(
                shape=(5, 9, 1), dtype="uint8", num_classes=5
            ),
            "segmentation2": Segmentation(
                shape=(5, 9, 1), dtype="uint8", names=("apple", "orange", "pineapple")
            ),
            "sequence": Sequence(
                dtype=Tensor(shape=(None, None), max_shape=(100, 100), dtype="uint8"),
            ),
            "text": Text((None,), max_shape=(10,)),
            "video": Video((100, 100, 3, 10)),
            "prim": Primitive("uint16", chunks=5, compressor="zstd"),
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


def test_serialize_error():
    with pytest.raises(TypeError):
        serialize([])
    with pytest.raises(TypeError):
        serialize({})


if __name__ == "__main__":
    test_serialize_deserialize()
    test_serialize_error()
