DEFAULT_GENERIC_TENSOR_PARAMS = {"htype": "generic", "sample_compression": None}
DEFAULT_IMAGE_TENSOR_PARAMS = {"htype": "image", "sample_compression": "jpeg"}

# Contains default kwargs for the tensors created from each of COCO keys
DEFAULT_COCO_TENSOR_PARAMS = {
    "segmentation": {
        "htype": "polygon",
        "sample_compression": None,
    },
    "category_id": {"htype": "class_label", "sample_compression": None},
    "bbox": {"htype": "bbox", "sample_compression": None},
    "_other": DEFAULT_GENERIC_TENSOR_PARAMS,
}
