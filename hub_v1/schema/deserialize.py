"""
License:
This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

from hub_v1.schema.features import Tensor, SchemaDict
from hub_v1.schema.image import Image
from hub_v1.schema.class_label import ClassLabel
from hub_v1.schema.polygon import Polygon
from hub_v1.schema.audio import Audio
from hub_v1.schema.bbox import BBox
from hub_v1.schema.mask import Mask
from hub_v1.schema.segmentation import Segmentation
from hub_v1.schema.sequence import Sequence
from hub_v1.schema.video import Video
from hub_v1.schema.text import Text


def _get_compressor(inp):
    return inp.get("compressor") or "default"


def deserialize(inp):
    if isinstance(inp, dict):
        if inp["type"] == "Audio":
            return Audio(
                shape=tuple(inp["shape"]),
                dtype=deserialize(inp["dtype"]),
                file_format=inp["file_format"],
                sample_rate=inp["sample_rate"],
                max_shape=tuple(inp["max_shape"]),
                chunks=inp["chunks"],
                compressor=_get_compressor(inp),
            )
        elif inp["type"] == "BBox":
            return BBox(
                shape=tuple(inp["shape"]),
                dtype=deserialize(inp["dtype"]),
                chunks=inp["chunks"],
                compressor=_get_compressor(inp),
                max_shape=tuple(inp["max_shape"]),
            )
        elif inp["type"] == "ClassLabel":
            if inp["_names"] is not None:
                return ClassLabel(
                    shape=tuple(inp["shape"]),
                    dtype=deserialize(inp["dtype"]),
                    names=inp["_names"],
                    chunks=inp["chunks"],
                    compressor=_get_compressor(inp),
                    max_shape=tuple(inp["max_shape"]),
                )
            else:
                return ClassLabel(
                    shape=tuple(inp["shape"]),
                    dtype=deserialize(inp["dtype"]),
                    num_classes=inp["_num_classes"],
                    chunks=inp["chunks"],
                    compressor=_get_compressor(inp),
                    max_shape=tuple(inp["max_shape"]),
                )
        elif inp["type"] == "SchemaDict" or inp["type"] == "FeatureDict":
            d = {}
            for k, v in inp["items"].items():
                d[k] = deserialize(v)
            return SchemaDict(d)
        elif inp["type"] == "Image":
            return Image(
                shape=tuple(inp["shape"]),
                dtype=deserialize(inp["dtype"]),
                max_shape=tuple(inp["max_shape"]),
                chunks=inp["chunks"],
                compressor=_get_compressor(inp),
            )
        elif inp["type"] == "Mask":
            return Mask(
                shape=tuple(inp["shape"]),
                max_shape=tuple(inp["max_shape"]),
                chunks=inp["chunks"],
                compressor=_get_compressor(inp),
            )
        elif inp["type"] == "Polygon":
            return Polygon(
                shape=tuple(inp["shape"]),
                max_shape=tuple(inp["max_shape"]),
                dtype=deserialize(inp["dtype"]),
                chunks=inp["chunks"],
                compressor=_get_compressor(inp),
            )
        elif inp["type"] == "Segmentation":
            class_labels = deserialize(inp["class_labels"])
            if class_labels._names is not None:
                return Segmentation(
                    shape=tuple(inp["shape"]),
                    dtype=deserialize(inp["dtype"]),
                    names=class_labels._names,
                    max_shape=tuple(inp["max_shape"]),
                    chunks=inp["chunks"],
                    compressor=_get_compressor(inp),
                )
            else:
                return Segmentation(
                    shape=tuple(inp["shape"]),
                    dtype=deserialize(inp["dtype"]),
                    num_classes=class_labels._num_classes,
                    max_shape=tuple(inp["max_shape"]),
                    chunks=inp["chunks"],
                    compressor=_get_compressor(inp),
                )
        elif inp["type"] == "Sequence":
            return Sequence(
                shape=tuple(inp["shape"]),
                dtype=deserialize(inp["dtype"]),
                max_shape=tuple(inp["max_shape"]),
                chunks=inp["chunks"],
                compressor=_get_compressor(inp),
            )
        elif inp["type"] == "Tensor":
            return Tensor(
                tuple(inp["shape"]),
                deserialize(inp["dtype"]),
                max_shape=tuple(inp["max_shape"]),
                chunks=inp["chunks"],
                compressor=_get_compressor(inp),
            )
        elif inp["type"] == "Text":
            return Text(
                tuple(inp["shape"]),
                deserialize(inp["dtype"]),
                max_shape=tuple(inp["max_shape"]),
                chunks=inp["chunks"],
                compressor=_get_compressor(inp),
            )
        elif inp["type"] == "Video":
            return Video(
                shape=tuple(inp["shape"]),
                dtype=deserialize(inp["dtype"]),
                # TODO uncomment back when image encoding will be added
                # encoding_format=inp["encoding_format"],
                max_shape=tuple(inp["max_shape"]),
                chunks=inp["chunks"],
                compressor=_get_compressor(inp),
            )
    else:
        return inp
