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


def deserialize(inp):
    if not isinstance(inp, dict):
        return inp
    if inp["type"] == "Audio":
        return Audio(
            shape=inp["shape"],
            dtype=deserialize(inp["dtype"]),
            file_format=inp["file_format"],
            sample_rate=inp["sample_rate"],
            max_shape=inp["max_shape"],
            chunks=inp["chunks"],
        )
    elif inp["type"] == "BBox":
        return BBox(dtype=deserialize(inp["dtype"]), chunks=inp["chunks"])
    elif inp["type"] == "ClassLabel":
        if "_num_classes" in inp.keys():
            return ClassLabel(num_classes=inp["_num_classes"], chunks=inp["chunks"])
        else:
            return ClassLabel(names=inp["names"], chunks=inp["chunks"])
    elif inp["type"] == "FeatureDict":
        d = {k: deserialize(v) for k, v in inp["items"].items()}
        return FeatureDict(d)
    elif inp["type"] == "Image":
        return Image(
            shape=tuple(inp["shape"]),
            dtype=deserialize(inp["dtype"]),
            encoding_format=inp["encoding_format"],
            max_shape=inp["max_shape"],
            chunks=inp["chunks"],
        )
    elif inp["type"] == "Mask":
        return Mask(
            shape=inp["shape"],
            dtype=deserialize(inp["dtype"]),
            max_shape=inp["max_shape"],
            chunks=inp["chunks"],
        )
    elif inp["type"] == "Polygon":
        return Polygon(
            shape=tuple(inp["shape"]),
            max_shape=inp["max_shape"],
            chunks=inp["chunks"],
        )
    elif inp["type"] == "Segmentation":
        class_labels = deserialize(inp["class_labels"])
        if hasattr(class_labels, "_num_classes"):
            return Segmentation(
                shape=inp["shape"],
                dtype=deserialize(inp["dtype"]),
                num_classes=class_labels._num_classes,
                max_shape=inp["max_shape"],
                chunks=inp["chunks"],
            )
        else:
            return Segmentation(
                shape=inp["shape"],
                dtype=deserialize(inp["dtype"]),
                names=class_labels.names,
                max_shape=inp["max_shape"],
                chunks=inp["chunks"],
            )
    elif inp["type"] == "Sequence":
        return Sequence(
            Tensor(shape=None, dtype=deserialize(inp["dtype"])),
            length=inp["shape"][0],
            chunks=inp["chunks"],
        )
    elif inp["type"] == "Tensor":
        return Tensor(
            tuple(inp["shape"]),
            deserialize(inp["dtype"]),
            max_shape=inp["max_shape"],
            chunks=inp["chunks"],
        )
    elif inp["type"] == "Video":
        return Video()
