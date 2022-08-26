from typing import Any, Callable, Optional, Sequence, Union, Type
from hub.core.dataset import Dataset


def clean_labels(
    dataset: Type[Dataset],
    # dataset_valid: Optional[Type[Dataset]] = None,
    module: Union[Any, Callable, None] = None,
    criterion: Optional[Any] = None,
    optimizer: Optional[Any] = None,
    optimizer_lr: int = 0.01,
    device: Union[str, Any, None] = None,
    epochs: int = 10,
    folds: int = 5,
    tensors: Optional[Sequence[str]] = None,
    dataloader_train_params: Optional[dict] = None,
    dataloader_valid_params: Optional[dict] = None,
    create_tensors: bool = False,
    overwrite: bool = False,
    verbose: bool = True,
):
    """
    Finds label errors in a dataset with cleanlab (github.com/cleanlab) open-source library.

    Note:
        Currently, only image classification tasks is supported. Therefore, the method accepts two tensors for the images and labels (e.g. `['images', 'labels']`).
        The tensors can be specified in `dataloader_train_params` in `transofrm` or `tensors`. Any PyTorch module can be used as a classifier.

    Args:
        module (class): A PyTorch torch.nn.Module module (class or instance). Default is `torchvision.models.resnet18()`.
        criterion (class): An uninitialized PyTorch criterion (loss) used to optimize the module. Default is `torch.nn.CrossEntropyLoss`.
        optimizer (class): An uninitialized PyTorch optimizer used to optimize the module. Default is `torch.optim.SGD`.
        optimizer_lr (int): The learning rate passed to the optimizer. Default is 0.01.
        device (str, torch.device): The compute device to be used. Default is `'cuda:0'` if available, else `'cpu'`.
        fold (int): Sets the number of cross-validation folds used to compute out-of-sample probabilities for each example in the dataset. The default is 5.
        epochs (int): The number of epochs to train for each `fit()` call. Default is 10.
        tensors (list): A list of tensor names that would be considered for cleaning (e.g. `['images', 'labels']`).
        dataloader_train_params (dict): Keyword arguments to pass into torch.utils.data.DataLoader. Options that may especially impact accuracy include: `shuffle`, `batch_size`.
        dataloader_valid_params (dict): Keyword arguments to pass into torch.utils.data.DataLoader. Options that may especially impact accuracy include: `shuffle`, `batch_size`. If not provided, `dataloader_train_params` will be used with `shuffle = False`.
        create_tensors (bool): if True, will create tensors `is_label_issue` and `label_quality_scores` under `label_issues group`. This would only work if you have write access to the dataset. Default is False.
        overwrite (bool): If True, will overwrite label_issues tensors if they already exists. Only applicable if `create_tensors` is True. Default is False.
        verbose (bool): This parameter controls how much output is printed. Default is True.

    Returns:
        label_issues: A boolean mask for the entire dataset where True represents a label issue and False represents an example that is confidently/accurately labeled.
        label_quality_scores: Returns label quality scores for each datapoint, where lower scores indicate labels less likely to be correct.

    Raises:
        ...

    """

    from hub.integrations.cleanlab import get_label_issues
    from hub.integrations.cleanlab import create_label_issues_tensors

    # TODO: Check if dataset is Hub Dataset
    # hub.core.dataset.hub_cloud_dataset.HubCloudDataset

    label_issues, label_quality_scores = get_label_issues(
        dataset=dataset,
        module=module,
        criterion=criterion,
        optimizer=optimizer,
        optimizer_lr=optimizer_lr,
        device=device,
        epochs=epochs,
        folds=folds,
        tensors=tensors,
        dataloader_train_params=dataloader_train_params,
        dataloader_valid_params=dataloader_valid_params,
        verbose=verbose,
    )

    if create_tensors:
        create_label_issues_tensors(
            dataset=dataset,
            label_issues=label_issues,
            label_quality_scores=label_quality_scores,
            overwrite=overwrite,
            verbose=verbose,
        )

    return label_issues, label_quality_scores
