from hub.util.exceptions import CheckoutError

import numpy as np


def is_subsettable(dataset, mask):
    """Returns True if dataset is subsettable"""
    return len(mask) == len(dataset)


def subset_dataset(dataset, mask):
    """Extracts subset of data examples where mask (np.ndarray) is True"""
    if not is_subsettable:
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


def process_label_issues(dataset, label_issues):

    columns = ["is_label_issue", "label_quality", "predicted_labels"]

    for column in columns:
        if column not in label_issues.columns:
            raise ValueError(
                f"DataFrame label_issues must contain column: `{column}`. "
            )

        if not is_subsettable(dataset=dataset, mask=label_issues[column]):
            raise ValueError(
                f"`{column}` is not a subset of the dataset. `{column}` and dataset must have same length"
            )

    if label_issues["is_label_issue"].dtype is not np.dtype("bool"):
        raise ValueError("`is_label_issue` must be a 1D np.ndarray with dtype `bool`.")

    if label_issues["label_quality"].dtype is not np.dtype("float64"):
        raise ValueError(
            "`label_quality` must be a 1D np.ndarray with dtype `float64`."
        )

    if label_issues["predicted_labels"].dtype is not np.dtype("int64"):
        raise ValueError(
            "`predicted_labels` must be a 1D np.ndarray with dtype `int64`."
        )

    return (
        label_issues["is_label_issue"].to_numpy(),
        label_issues["label_quality"].to_numpy(),
        label_issues["predicted_labels"].to_numpy(),
    )
