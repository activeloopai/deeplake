from typing import Any, Optional, Union
import pandas as pd


def clean_labels(
    dataset: Any,
    model: Any,
    folds: int = 5,
    verbose: bool = True,
    label_issues_kwargs: Optional[dict] = {},
    label_quality_kwargs: Optional[dict] = {},
):
    """
    Finds label errors in a dataset with cleanlab (github.com/cleanlab) open-source library.

    Note:
        Currently, only image classification tasks is supported. Therefore, the method accepts two tensors for the images and labels (e.g. `['images', 'labels']`).
        The tensors can be specified in `transofrm` or `tensors`. Any PyTorch module can be used as a classifier.

    Args:
        dataset (class): Hub Dataset for training. The label issues will be computed for training set.
        model (class): An instantiated skorch NeuralNet module.
        folds (int): Sets the number of cross-validation folds used to compute out-of-sample probabilities for each example in the dataset. The default is 5.
        verbose (bool): This parameter controls how much output is printed. Default is True.
        label_issues_kwargs (dict, Optional): Keyword arguments to be passed to the `cleanlab.filter.find_label_issues` function. Options that may especially impact accuracy include: filter_by, frac_noise, min_examples_per_class. Default is `None`.
        label_quality_kwargs (dict, Optional): Keyword arguments to be passed to the `cleanlab.rank.get_label_quality_scores` function. Options include: method, adjust_pred_probs. Default is `None`.

    Returns:
        pandas DataFrame of label issues for each example. Each row represents an example from the dataset and the DataFrame contains the following columns:
            - label_issues: A boolean mask for the entire dataset where True represents a label issue and False represents an example that is confidently/accurately labeled.
            - label_quality_scores: Label quality scores for each datapoint, where lower scores indicate labels less likely to be correct.
            - predicted_labels: Class predicted by model trained on cleaned data for each example in the dataset.

    Raises:
        ...

    """

    from hub.integrations.cleanlab.label_issues import get_label_issues
    from hub.integrations.utils import is_hub_dataset

    # Catch most common user errors early.
    if not is_hub_dataset(dataset):
        raise TypeError(f"`dataset` must be a Hub Dataset. Got {type(dataset)}")

    label_issues, label_quality_scores, predicted_labels = get_label_issues(
        dataset=dataset,
        model=model,
        folds=folds,
        verbose=verbose,
        label_issues_kwargs=label_issues_kwargs,
        label_quality_kwargs=label_quality_kwargs,
    )

    return pd.DataFrame(
        {
            "is_label_issue": label_issues,
            "label_quality": label_quality_scores,
            "predicted_labels": predicted_labels,
        }
    )


def create_tensors(
    dataset: Any,
    label_issues: Any,
    branch: Union[str, None] = None,
    overwrite: bool = False,
    verbose: bool = True,
):
    """
    Creates tensors `is_label_issue` and `label_quality_scores` and `predicted_labels` under `label_issues group`.

    Note:
        This method would only work if you have write access to the dataset.

    Args:
        dataset (class): Hub Dataset to add the tensors to.
        label_issues (class): pandas DataFrame of label issues for each example computed by running `clean_labels()`.
        branch (str, Optional): The name of the branch to use for creating the label_issues tensor group. If the branch name is provided but the branch does not exist, it will be created. If no branch is provided, the default branch will be used.
        overwrite (bool): If True, will overwrite label_issues tensors if they already exists. Only applicable if `create_tensors` is True. Default is False.
        verbose (bool): This parameter controls how much output is printed. Default is True.

    Returns:
        commit_id (str): The commit hash of the commit that was created.

    Raises:
        ...

    """
    from hub.integrations.cleanlab.tensors import create_label_issues_tensors
    from hub.integrations.cleanlab.utils import switch_branch
    from hub.integrations.utils import is_hub_dataset

    if not is_hub_dataset(dataset):
        raise TypeError(f"`dataset` must be a Hub Dataset. Got {type(dataset)}")

    # Catch write access error early.
    if dataset.read_only:
        raise ValueError(
            f"`create_tensors` is True but dataset is read-only. Try loading the dataset with `read_only=False.`"
        )

    if branch:
        switch_branch(dataset=dataset, branch=branch)

    if verbose:
        print(
            f"The `label_issues` tensor will be committed to {dataset.branch} branch."
        )

    commit_id = create_label_issues_tensors(
        dataset=dataset,
        label_issues=label_issues,
        overwrite=overwrite,
        verbose=verbose,
    )

    return commit_id


def clean_view(dataset: Any, label_issues: Optional[Any] = None):
    """
    Returns a view of the dataset with clean labels.

    Note:
        If `label_issues` np.ndarray is not provided, the function will check if the dataset has a `label_issues/is_label_issue` tensor. If so, the function will use it to filter the dataset.

    Args:
        dataset (class): Hub Dataset to be used to get a flitered view.
        label_issues (np.ndarray, Optional): A boolean mask for the entire dataset where True represents a label issue and False represents an example that is accurately labeled. Default is `None`.

    Returns:
        cleaned_dataset (class): Dataset view where only clean labels are present, and the rest are filtered out.

    """
    from hub.integrations.cleanlab.utils import subset_dataset, process_label_issues

    if label_issues is not None:
        label_issues, _, _ = process_label_issues(label_issues)
        label_issues_mask = ~label_issues

    # If label_issues is not provided as user input, try to get it from the tensor.
    elif "label_issues/is_label_issue" in dataset.tensors:
        label_issues_mask = ~dataset.label_issues.is_label_issue.numpy()

    else:
        raise ValueError(
            "No `label_issues/is_label_issue` tensor found and no `label_issues` np.ndarray provided. Please run `clean_labels` first to obtain `label_issues` boolean mask."
        )

    cleaned_dataset = subset_dataset(dataset=dataset, mask=label_issues_mask)

    return cleaned_dataset
