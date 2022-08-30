from typing import Any, Callable, Optional, Sequence, Union, Type
from hub.core.dataset import Dataset


def clean_labels(
    dataset: Type[Dataset],
    dataset_valid: Optional[Type[Dataset]] = None,
    transform: Optional[Callable] = None,
    tensors: Optional[Sequence[str]] = None,
    batch_size: int = 64,
    module: Union[Any, Callable, None] = None,
    criterion: Optional[Any] = None,
    optimizer: Optional[Any] = None,
    optimizer_lr: int = 0.01,
    device: Union[str, Any, None] = None,
    epochs: int = 10,
    shuffle: bool = False,
    folds: int = 5,
    verbose: bool = True,
    skorch_kwargs: Optional[dict] = None,
):
    """
    Finds label errors in a dataset with cleanlab (github.com/cleanlab) open-source library.

    Note:
        Currently, only image classification tasks is supported. Therefore, the method accepts two tensors for the images and labels (e.g. `['images', 'labels']`).
        The tensors can be specified in `transofrm` or `tensors`. Any PyTorch module can be used as a classifier.

    Args:
        dataset (class): Hub Dataset for training. The label issues will be computed for training set.
        dataset_valid (class, Optional): Hub Dataset to use as a validation set for training. The label issues will not be computed for this set.
        It is assumed that the validation tensor names are the same as the training tensor names. Default is `None`.
        transform (Callable, Optional): Transformation function to be applied to each sample. Default is `None`.
        tensors (list, Optional): A list of two tensors (in the following order: data, labels) that would be used to find label issues (e.g. `['images', 'labels']`).
        batch_size (int): Number of samples per batch to load. If `batch_size` is -1, a single batch with all the data will be used during training and validation. Default is `64`.
        module (class): A PyTorch torch.nn.Module module (class or instance). Default is `torchvision.models.resnet18()`.
        criterion (class): An uninitialized PyTorch criterion (loss) used to optimize the module. Default is `torch.nn.CrossEntropyLoss`.
        optimizer (class): An uninitialized PyTorch optimizer used to optimize the module. Default is `torch.optim.SGD`.
        optimizer_lr (int): The learning rate passed to the optimizer. Default is 0.01.
        device (str, torch.device): The compute device to be used. Default is `'cuda:0'` if available, else `'cpu'`.
        epochs (int): The number of epochs to train for each `fit()` call. Note that you may keyboard-interrupt training at any time. Default is 10.
        shuffle (bool): Whether to shuffle the data before each epoch. Default is `False`.
        folds (int): Sets the number of cross-validation folds used to compute out-of-sample probabilities for each example in the dataset. The default is 5.
        verbose (bool): This parameter controls how much output is printed. Default is True.
        skorch_kwargs (dict, Optional): Keyword arguments to be passed to the skorch module (skorch.readthedocs.io/en/stable/net.html).
        Additionally, `iterator_train__transform` and iterator_valid__transform` can be used to set params for the training and validation iterators. Default is `None`.

    Returns:
        label_issues (np.ndarray): A boolean mask for the entire dataset where True represents a label issue and False represents an example that is confidently/accurately labeled.
        label_quality_scores (np.ndarray): Label quality scores for each datapoint, where lower scores indicate labels less likely to be correct.
        predicted_labels (np.ndarray): Class predicted by model trained on cleaned data for each example in the dataset.

    Raises:
        ...

    """

    from hub.integrations.cleanlab import get_label_issues
    from hub.integrations.cleanlab.utils import is_dataset

    # Catch most common user errors early.
    if not is_dataset(dataset):
        raise TypeError(f"`dataset` must be a Hub Dataset. Got {type(dataset)}")

    if dataset_valid and not is_dataset(dataset_valid):
        raise TypeError(
            f"`dataset_valid` must be a Hub Dataset. Got {type(dataset_valid)}"
        )

    label_issues, label_quality_scores, predicted_labels = get_label_issues(
        dataset=dataset,
        dataset_valid=dataset_valid,
        transform=transform,
        tensors=tensors,
        batch_size=batch_size,
        module=module,
        criterion=criterion,
        optimizer=optimizer,
        optimizer_lr=optimizer_lr,
        device=device,
        epochs=epochs,
        shuffle=shuffle,
        folds=folds,
        verbose=verbose,
        skorch_kwargs=skorch_kwargs,
    )

    return label_issues, label_quality_scores, predicted_labels


def create_tensors(
    dataset: Type[Dataset],
    label_issues: Any,
    label_quality_scores: Any,
    predicted_labels: Any,
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
        label_issues (np.ndarray): A boolean mask for the entire dataset where True represents a label issue and False represents an example that is confidently/accurately labeled.
        label_quality_scores (np.ndarray): Label quality scores for each datapoint, where lower scores indicate labels less likely to be correct.
        predicted_labels (np.ndarray): Class predicted by model trained on cleaned data for each example in the dataset.
        branch (str, Optional): The name of the branch to use for creating the label_issues tensor group. If the branch name is provided but the branch does not exist, it will be created.
        If no branch is provided, the default branch will be used.
        overwrite (bool): If True, will overwrite label_issues tensors if they already exists. Only applicable if `create_tensors` is True. Default is False.
        verbose (bool): This parameter controls how much output is printed. Default is True.

    Returns:
        commit_id (str): The commit hash of the commit that was created.

    Raises:
        ...

    """
    from hub.integrations.cleanlab import create_label_issues_tensors
    from hub.integrations.cleanlab.utils import is_dataset, switch_branch

    if not is_dataset(dataset):
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
        label_quality_scores=label_quality_scores,
        predicted_labels=predicted_labels,
        overwrite=overwrite,
        verbose=verbose,
    )

    return commit_id


def clean_view(dataset: Type[Dataset], label_issues: Optional[Any] = None):
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
    from hub.integrations.cleanlab.utils import subset_dataset, assert_label_issues

    if label_issues is not None:
        label_issues_mask = ~label_issues

    # If label_issues is not provided as user input, try to get it from the tensor.
    elif "label_issues/is_label_issue" in dataset.tensors:
        label_issues_mask = ~dataset.label_issues.is_label_issue.numpy()

    else:
        raise ValueError(
            "No `label_issues/is_label_issue` tensor found and no `label_issues` np.ndarray provided. Please run `clean_labels` first to obtain `label_issues` boolean mask."
        )

    assert_label_issues(dataset=dataset, label_issues=label_issues_mask)
    cleaned_dataset = subset_dataset(dataset=dataset, mask=label_issues_mask)

    return cleaned_dataset
