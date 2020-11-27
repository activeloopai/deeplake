from hub.features.features import Tensor, FeatureDict
from hub.features.image import Image
from hub.features.class_label import ClassLabel
from hub.features.polygon import Polygon
from hub.features.audio import Audio
from hub.features.bbox import BBox
from hub.features.mask import Mask
from hub.features.segmentation import Segmentation
from hub.features.sequence import Sequence
from hub.features.video import Video
from hub.features.text import Text


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
                max_shape=inp["max_shape"],
                chunks=inp["chunks"],
                compressor=_get_compressor(inp),
            )
        elif inp["type"] == "BBox":
            return BBox(
                dtype=deserialize(inp["dtype"]),
                chunks=inp["chunks"],
                compressor=_get_compressor(inp),
            )
        elif inp["type"] == "ClassLabel":
            if "_num_classes" in inp.keys():
                return ClassLabel(
                    num_classes=inp["_num_classes"],
                    chunks=inp["chunks"],
                    compressor=_get_compressor(inp)
                )
            else:
                return ClassLabel(
                    names=inp["names"],
                    chunks=inp["chunks"],
                    compressor=_get_compressor(inp)
                )
        elif inp["type"] == "FeatureDict":
            d = {}
            for k, v in inp["items"].items():
                d[k] = deserialize(v)
            return FeatureDict(d)
        elif inp["type"] == "Image":
            return Image(
                shape=tuple(inp["shape"]),
                dtype=deserialize(inp["dtype"]),
                # TODO uncomment back when image encoding will be added
                # encoding_format=inp["encoding_format"],
                max_shape=inp["max_shape"],
                chunks=inp["chunks"],
                compressor=_get_compressor(inp),
            )
        elif inp["type"] == "Mask":
            return Mask(
                shape=tuple(inp["shape"]),
                max_shape=inp["max_shape"],
                chunks=inp["chunks"],
                compressor=_get_compressor(inp),
            )
        elif inp["type"] == "Polygon":
            return Polygon(
                shape=tuple(inp["shape"]),
                max_shape=inp["max_shape"],
                dtype=deserialize(inp["dtype"]),
                chunks=inp["chunks"],
                compressor=_get_compressor(inp),
            )
        elif inp["type"] == "Segmentation":
            class_labels = deserialize(inp["class_labels"])
            if hasattr(class_labels, "_num_classes"):
                return Segmentation(
                    shape=tuple(inp["shape"]),
                    dtype=deserialize(inp["dtype"]),
                    num_classes=class_labels._num_classes,
                    max_shape=inp["max_shape"],
                    chunks=inp["chunks"],
                    compressor=_get_compressor(inp),
                )
            else:
                return Segmentation(
                    shape=tuple(inp["shape"]),
                    dtype=deserialize(inp["dtype"]),
                    names=class_labels.names,
                    max_shape=inp["max_shape"],
                    chunks=inp["chunks"],
                    compressor=_get_compressor(inp),
                )
        elif inp["type"] == "Sequence":
            return Sequence(
                shape=tuple(inp["shape"]),
                dtype=deserialize(inp["dtype"]),
                max_shape=inp["max_shape"],
                chunks=inp["chunks"],
                compressor=_get_compressor(inp)
            )
        elif inp["type"] == "Tensor":
            return Tensor(
                tuple(inp["shape"]),
                deserialize(inp["dtype"]),
                max_shape=inp["max_shape"],
                chunks=inp["chunks"],
                compressor=_get_compressor(inp),
            )
        elif inp["type"] == "Text":
            return Text(
                tuple(inp["shape"]),
                deserialize(inp["dtype"]),
                max_shape=inp["max_shape"],
                chunks=inp["chunks"],
                compressor=_get_compressor(inp),
            )
        elif inp["type"] == "Video":
            return Video(
                shape=tuple(inp["shape"]),
                dtype=deserialize(inp["dtype"]),
                # TODO uncomment back when image encoding will be added
                # encoding_format=inp["encoding_format"],
                max_shape=inp["max_shape"],
                chunks=inp["chunks"],
                compressor=_get_compressor(inp),
            )
    else:
        return inp
