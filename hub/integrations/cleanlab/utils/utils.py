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
    return len(mask) == len(dataset)


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


def is_valid_ndarray(dataset, array):
    """Returns True if array is a 1D np.ndarray with length equal to the dataset size"""
    return isinstance(array, np.ndarray) and len(array) == len(dataset)


def assert_label_issues(dataset, label_issues):
    """Asserts that label_issues is a 1D np.ndarray with dtype bool and length equal to the dataset size"""
    if not is_valid_ndarray(dataset=dataset, array=label_issues):
        raise ValueError(
            "`label_issues` must be a 1D np.ndarray with length equal to the dataset size."
        )
    if label_issues.dtype is not np.dtype("bool"):
        raise ValueError("`label_issues` must be a 1D np.ndarray with dtype `bool`.")


def assert_label_quality_scores(dataset, label_quality_scores):
    """Asserts that label_quality_scores is a 1D np.ndarray with dtype bool and length equal to the dataset size"""
    if not is_valid_ndarray(dataset=dataset, array=label_quality_scores):
        raise ValueError(
            "`label_quality_scores` must be a 1D np.ndarray with length equal to the dataset size."
        )

    if label_quality_scores.dtype is not np.dtype("float64"):
        raise ValueError(
            "`label_quality_scores` must be a 1D np.ndarray with dtype `float64`."
        )


def assert_predicted_labels(dataset, predicted_labels):
    """Asserts that predicted_labels is a 1D np.ndarray with dtype bool and length equal to the dataset size"""
    if not is_valid_ndarray(dataset=dataset, array=predicted_labels):
        raise ValueError(
            "`label_quality_scores` must be a 1D np.ndarray with length equal to the dataset size."
        )

    if predicted_labels.dtype is not np.dtype("int64"):
        raise ValueError(
            "`predicted_labels` must be a 1D np.ndarray with dtype `int64`."
        )
