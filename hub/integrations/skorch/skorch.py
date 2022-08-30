import torch

from torchvision.models import resnet18

from skorch.helper import predefined_split

from .net import VisionClassifierNet

from typing import Any, Callable, Optional, Sequence, Union, Type

# from hub.core.dataset import Dataset
import numpy as np

def pytorch_module_to_skorch(
    dataset: Any,
    dataset_valid: Any = None,
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
    This function wraps a PyTorch Module in a skorch NeuralNet. It will also initialize a default PyTorch module if one is not provided.

    Args:
        dataset (class): Hub Dataset to use to instantiate the NeuralNet.
        dataset_valid (class, Optional): Hub Dataset to use as a validation set for training. It is expected that the validation set tensor names are the same as the training tensor names. Default is `None`.
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
        skorch_kwargs (dict, Optional): Keyword arguments to be passed to the skorch `NeuralNet` constructor.  Additionally,`iterator_train__transform` and iterator_valid__transform` can be used to set params for the training and validation iterators. Default is {}.

    Returns:
        model (class): A skorch NeuralNet instance.

    """
    from hub.integrations.skorch.utils import repeat_image_shape, get_dataset_tensors

    # if dataset_valid and not is_dataset(dataset_valid):
    #     raise TypeError(
    #         f"`dataset_valid` must be a Hub Dataset. Got {type(dataset_valid)}"
    #     )

    images_tensor, labels_tensor = get_dataset_tensors(
        dataset=dataset,
        transform=transform,
        tensors=tensors,
    )

    images_tensor, labels_tensor = tensors

    if device is None:
        if torch.cuda.is_available():
            device_name = "cuda:0"
        # elif torch.backends.mps.is_available():
        #     device_name = "mps"
        else:
            device_name = "cpu"
        device = torch.device(device_name)

    if module is None:
        # Set default module.
        module = resnet18()

        # Make training work with both grayscale and color images.
        transform = repeat_image_shape(images_tensor, transform)

        # Change the last layer to have num_classes output channels.
        labels = dataset[labels_tensor].numpy().flatten()
        num_classes = len(np.unique(labels))
        module.fc = torch.nn.Linear(module.fc.in_features, num_classes)

    if criterion is None:
        criterion = torch.nn.CrossEntropyLoss

    if optimizer is None:
        optimizer = torch.optim.Adam

    if dataset_valid:
        train_split = predefined_split(dataset_valid)
    else:
        train_split = None

    model = VisionClassifierNet(
        module=module,
        batch_size=batch_size,
        criterion=criterion,
        device=device,
        max_epochs=epochs,
        optimizer=optimizer,
        optimizer__lr=optimizer_lr,
        train_split=train_split,
        images_tensor=images_tensor,
        labels_tensor=labels_tensor,
        iterator_train__shuffle=shuffle,
        iterator_train__transform=transform,
        iterator_valid__transform=transform,
    )

    # Set optional kwargs params for the neural net. This will override any params set in the module.
    model.set_params(**skorch_kwargs)

    return model
