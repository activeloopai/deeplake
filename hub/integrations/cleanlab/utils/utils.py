from hub.core.dataset import Dataset


def is_dataset(dataset):
    return isinstance(dataset, Dataset)


def is_image_tensor(image_tensor_htype):
    unsupported_image_htypes = set(
        "bbox",
        "point",
        "video",
        "audio",
        "segment_mask",
        "binary_mask",
        "keypoints_coco",
        "class_label",
    )
    return (
        image_tensor_htype in unsupported_image_htypes
        or image_tensor_htype.startswith("sequence")
    )


def is_label_tensor(label_tensor_htype):
    unsupported_label_htypes = set(
        "bbox",
        "point",
        "video",
        "audio",
        "segment_mask",
        "binary_mask",
        "keypoints_coco",
        "image",
    )
    return (
        label_tensor_htype in unsupported_label_htypes
        or label_tensor_htype.startswith("sequence")
    )
