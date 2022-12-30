import torch

from torchvision.models import resnet18

from skorch.helper import predefined_split

from hub.integrations.skorch.net import VisionClassifierNet


def pytorch_module_to_skorch(
    dataset,
    validation_dataset,
    transform,
    tensors,
    batch_size,
    module,
    criterion,
    device,
    epochs,
    shuffle,
    optimizer,
    optimizer_lr,
    skorch_kwargs,
):

    from hub.integrations.skorch.utils import get_dataset_tensors
    from hub.integrations.common.utils import get_num_classes, get_labels

    images_tensor, labels_tensor = get_dataset_tensors(
        dataset=dataset,
        transform=transform,
        tensors=tensors,
    )

    if device is None:
        if torch.cuda.is_available():
            device_name = "cuda:0"
        # TODO: add check if pytorch version is nightly
        # elif torch.backends.mps.is_available():
        #     device_name = "mps"
        else:
            device_name = "cpu"
        device = torch.device(device_name)

    if module is None:
        # Set default module.
        module = resnet18()

        # Change the last layer to have num_classes output channels.
        labels = get_labels(dataset=dataset, labels_tensor=labels_tensor)
        module.fc = torch.nn.Linear(module.fc.in_features, get_num_classes(labels))

    if criterion is None:
        criterion = torch.nn.CrossEntropyLoss

    if optimizer is None:
        optimizer = torch.optim.Adam

    if validation_dataset:
        train_split = predefined_split(validation_dataset)
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
