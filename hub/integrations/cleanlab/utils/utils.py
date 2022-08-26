from hub.core.dataset import Dataset


def is_dataset(dataset):
    return isinstance(dataset, Dataset)


def is_image_tensor(image_tensor_htype):
    supported_image_htypes = set(
        [
            "image",
            "image.rgb",
            "image.gray",
            "generic"

        ],
    )
    return (
        image_tensor_htype in supported_image_htypes
        and not image_tensor_htype.startswith("sequence")
    )


def is_label_tensor(label_tensor_htype):
    supported_label_htypes = set(
        [
            "class_label",
            "generic"
        ],
    )
    return (
        label_tensor_htype in supported_label_htypes
        and not label_tensor_htype.startswith("sequence")
    )
