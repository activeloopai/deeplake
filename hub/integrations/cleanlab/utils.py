from hub.util.exceptions import CheckoutError

import numpy as np


def is_subsettable(dataset, mask):
    """Returns True if dataset is subsettable"""
    return len(mask) == len(dataset)


def subset_dataset(dataset, mask):
    """Extracts subset of data examples where mask (np.ndarray) is True"""
    indices = np.where(mask)[0].tolist()
    return dataset[indices]


def switch_branch(dataset, branch):
    """Switches dataset to a different branch"""
    # If branch is provided, check if it exists. If not, create it.
    try:
        dataset.checkout(branch)
    except CheckoutError:
        dataset.checkout(branch, create=True)


def process_label_issues(dataset, label_issues):
    """This is a helper function to process the label_issues DataFrame into a tuple of numpy ndarrays."""
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

    if not label_issues["is_label_issue"].dtype in [np.dtype("bool"), np.dtype("int")]:
        raise ValueError("`is_label_issue` must be a 1D np.ndarray with dtype `bool`.")

    if not np.issubdtype(label_issues["label_quality"].dtype, np.floating):
        raise ValueError("`label_quality` must be a 1D np.ndarray with dtype `float`.")

    if not np.issubdtype(label_issues["predicted_labels"].dtype, np.integer):
        raise ValueError("`predicted_labels` must be a 1D np.ndarray with dtype `int`.")

    return (
        label_issues["is_label_issue"].to_numpy(),
        label_issues["label_quality"].to_numpy(),
        label_issues["predicted_labels"].to_numpy(),
    )
