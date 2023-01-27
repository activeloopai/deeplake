from typing import Dict
from deeplake.constants import UNSPECIFIED

DEFAULT_GENERIC_TENSOR_PARAMS = {"htype": UNSPECIFIED, "sample_compression": None}
DEFAULT_IMAGE_TENSOR_PARAMS = {
    "name": "images",
    "htype": "image",
}

# Contains default kwargs for the tensors created from each of COCO keys
DEFAULT_COCO_TENSOR_PARAMS: Dict[str, Dict] = {
    "segmentation": {
        "htype": "polygon",
        "sample_compression": None,
    },
    "category_id": {"htype": "class_label", "sample_compression": None},
    "bbox": {"htype": "bbox", "sample_compression": None},
    "keypoints": {"htype": "keypoints_coco"},
    "_other": DEFAULT_GENERIC_TENSOR_PARAMS,
}
