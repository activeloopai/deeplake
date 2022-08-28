from hub.core.dataset import Dataset

import numpy as np


def is_dataset(dataset):
    return isinstance(dataset, Dataset)


def is_np_ndarray(array):
    return isinstance(array, np.ndarray)


def is_image_tensor(image_tensor_htype):
    supported_image_htypes = set(
        ["image", "image.rgb", "image.gray", "generic"],
    )
    return (
        image_tensor_htype in supported_image_htypes
        and not image_tensor_htype.startswith("sequence")
    )


def is_label_tensor(label_tensor_htype):
    supported_label_htypes = set(
        ["class_label", "generic"],
    )
    return (
        label_tensor_htype in supported_label_htypes
        and not label_tensor_htype.startswith("sequence")
    )


def subset_dataset(dataset, mask):
    """Extracts subset of data examples where mask (np.ndarray) is True"""
    if is_np_ndarray(mask):
        mask = np.where(mask)[0].tolist()
    else:
        raise ValueError(f"Mask must be a 1D np.ndarray, got {type(mask)}")

    return dataset[mask]
