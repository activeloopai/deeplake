import torch

from torchvision.models import resnet18

from skorch import NeuralNet
from skorch.helper import predefined_split

# Wraps the PyTorch Module in a sklearn interface.
class VisionClassifierNet(NeuralNet):
    """
    This class extends the `NeuralNet` class from skorch.
    It overrides `get_dataset` and `get_iterator` to return the Hub's PyTorch Dataloader.
    Additionally, it overrides `train_step_single`, `evaluation_step` and `validation_step`
    to get the data for each batch from the images and label tensors.
    """

    def __init__(
        self,
        images_tensor,
        labels_tensor,
        **kwargs,
    ):
        super(VisionClassifierNet, self).__init__(**kwargs)
        self.images_tensor = images_tensor
        self.labels_tensor = labels_tensor

    def get_dataset(self, dataset, y=None):
        return dataset

    def get_iterator(self, dataset, training=False):
        if training:
            kwargs = self.get_params_for("iterator_train")

        else:
            kwargs = self.get_params_for("iterator_valid")

        if "batch_size" not in kwargs:
            kwargs["batch_size"] = self.batch_size

        if kwargs["batch_size"] == -1:
            kwargs["batch_size"] = len(dataset)

        return dataset.pytorch(**kwargs)

    def train_step_single(self, batch, **fit_params):
        self._set_training(True)
        Xi, yi = batch[self.images_tensor], torch.squeeze(batch[self.labels_tensor])

        y_pred = self.infer(Xi, **fit_params)
        loss = self.get_loss(y_pred, yi, X=Xi, training=True)
        loss.backward()
        return {
            "loss": loss,
            "y_pred": y_pred,
        }

    def evaluation_step(self, batch, training=False):
        self.check_is_fitted()
        Xi = batch[self.images_tensor]
        with torch.set_grad_enabled(training):
            self._set_training(training)
            return self.infer(Xi)

    def validation_step(self, batch, **fit_params):
        self._set_training(False)
        Xi, yi = batch[self.images_tensor], torch.squeeze(batch[self.labels_tensor])
        with torch.no_grad():
            y_pred = self.infer(Xi, **fit_params)
            loss = self.get_loss(y_pred, yi, X=Xi, training=False)
        return {
            "loss": loss,
            "y_pred": y_pred,
        }


def pytorch_module_to_skorch(
    dataset_valid,
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
    num_classes,
):
    """
    This function wraps a PyTorch Module in a skorch NeuralNet.
    It will also initialize a default PyTorch module if one is not provided.
    """
    from hub.integrations.skorch.utils import repeat_shape

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
        transform = repeat_shape(images_tensor, transform)

        # Change the last layer to have num_classes output channels.
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
    )

    # Set transoform params for the train and validation dataloader.
    model.set_params(
        iterator_train__transform=transform, iterator_valid__transform=transform
    )

    return model
