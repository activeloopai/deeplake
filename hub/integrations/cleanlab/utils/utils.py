from hub.core.dataset import Dataset
from hub.util.exceptions import CheckoutError

import numpy as np


def is_dataset(dataset):
    return isinstance(dataset, Dataset)


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


def is_dataset_subsettable(dataset, mask):
    """Returns True if dataset is subsettable"""
    return len(dataset) == len(mask)


def subset_dataset(dataset, mask):
    """Extracts subset of data examples where mask (np.ndarray) is True"""
    if not is_dataset_subsettable:
        raise ValueError(
            "`label_issues` mask is not a subset of the dataset. Please provide a mask that is a subset of the dataset."
        )

    try:
        # Extract indices where mask is True.
        indices = np.where(mask)[0].tolist()
    except Exception:
        raise TypeError(f"`label_issues` must be a 1D np.ndarray, got {type(mask)}")

    return dataset[indices]


def switch_branch(dataset, branch):
    """Switches dataset to a different branch"""
    # If branch is provided, check if it exists. If not, create it.
    try:
        dataset.checkout(branch)
    except CheckoutError:
        dataset.checkout(branch, create=True)
