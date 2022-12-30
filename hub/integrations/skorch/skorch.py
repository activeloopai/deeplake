from typing import Any, Callable, Optional, Sequence, Union


def skorch(
    dataset: Any,
    validation_dataset: Any = None,
    transform: Optional[Callable] = None,
    tensors: Optional[Sequence[str]] = None,
    batch_size: int = 64,
    module: Union[Any, Callable, None] = None,
    criterion: Optional[Any] = None,
    device: Union[str, Any, None] = None,
    epochs: int = 10,
    shuffle: bool = False,
    optimizer: Optional[Any] = None,
    optimizer_lr: int = 0.01,
    skorch_kwargs: Optional[dict] = {},
):
    """
    This function wraps a PyTorch Module in a skorch NeuralNet. It will also initialize a default PyTorch module if one is not provided. Any PyTorch module can be used as a classifier.

    Note:
        Currently, only image classification tasks is supported. Therefore, two tensors for the images and labels should be specified (e.g. `['images', 'labels']`).

    Args:
        dataset (class): Hub Dataset to use to instantiate the NeuralNet.
        validation_dataset (class, Optional): Hub Dataset to use as a validation set for training. It is expected that the validation set tensor names are the same as the training tensor names. Default is `None`.
        transform (Callable, Optional): Transformation function to be applied to each sample. This be used to provide ordered tensor names (data, labels). Default is `None`.
        tensors (list, Optional): A list of ordered tensors (data, labels) that would be used to find label issues (e.g. `['images', 'labels']`).
        batch_size (int): Number of samples per batch to load. If `batch_size` is -1, a single batch with all the data will be used during training and validation. Default is `64`.
        module (class): A PyTorch torch.nn.Module module (class or instance). Default is `torchvision.models.resnet18()`.
        criterion (class): An uninitialized PyTorch criterion (loss) used to optimize the module. Default is `torch.nn.CrossEntropyLoss`.
        optimizer (class): An uninitialized PyTorch optimizer used to optimize the module. Default is `torch.optim.SGD`.
        optimizer_lr (int): The learning rate passed to the optimizer. Default is 0.01.
        device (str, torch.device): The compute device to be used. Default is `'cuda:0'` if available, else `'cpu'`.
        epochs (int): The number of epochs to train for each `fit()` call. Note that you may keyboard-interrupt training at any time. Default is 10.
        shuffle (bool): Whether to shuffle the data before each epoch. Default is `False`.
        skorch_kwargs (dict, Optional): Keyword arguments to be passed to the skorch `NeuralNet` constructor. Additionally,`iterator_train__transform` and iterator_valid__transform` can be used to set params for the training and validation iterators. Default is {}.

    Returns:
        model (class): A skorch NeuralNet instance.

    """
    from hub.integrations.skorch.module import pytorch_module_to_skorch
    from hub.integrations.common.utils import is_dataset

    if not is_dataset(dataset):
        raise TypeError(f"`dataset` must be a Hub Dataset. Got {type(dataset)}")

    if validation_dataset and not is_dataset(validation_dataset):
        raise TypeError(
            f"`validation_dataset` must be a Hub Dataset. Got {type(validation_dataset)}"
        )

    return pytorch_module_to_skorch(
        dataset=dataset,
        validation_dataset=validation_dataset,
        transform=transform,
        tensors=tensors,
        batch_size=batch_size,
        module=module,
        criterion=criterion,
        device=device,
        epochs=epochs,
        shuffle=shuffle,
        optimizer=optimizer,
        optimizer_lr=optimizer_lr,
        skorch_kwargs=skorch_kwargs,
    )
