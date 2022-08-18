from skorch import NeuralNet
from torch import set_grad_enabled, squeeze, no_grad
from torch.nn import CrossEntropyLoss, Linear, Conv2d
from torchvision.models import resnet18
from torch.optim import Adam

# Wraps the PyTorch Module in an sklearn interface.
class VisionClassifierNet(NeuralNet):
    def __init__(
        self,
        dataloader_train_params,
        dataloader_valid_params,
        images_tensor,
        labels_tensor,
        **kwargs
    ):
        super(VisionClassifierNet, self).__init__(**kwargs)
        self.dataloader_train_params = dataloader_train_params
        self.dataloader_valid_params = dataloader_valid_params
        self.images_tensor = images_tensor
        self.labels_tensor = labels_tensor

    def get_dataset(self, dataset, y=None):
        return dataset

    def get_iterator(self, dataset, training=False):
        if training:
            kwargs = self.dataloader_train_params

        else:
            kwargs = self.dataloader_valid_params
            if kwargs is None:
                kwargs = self.dataloader_train_params
                kwargs["shuffle"] = False

        if "batch_size" not in kwargs:
            kwargs["batch_size"] = self.batch_size

        if kwargs["batch_size"] == -1:
            kwargs["batch_size"] = len(dataset)

        return dataset.pytorch(**kwargs)

    def train_step_single(self, batch, **fit_params):
        self._set_training(True)
        Xi, yi = batch[self.images_tensor], squeeze(batch[self.labels_tensor])

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
        with set_grad_enabled(training):
            self._set_training(training)
            return self.infer(Xi)

    def validation_step(self, batch, **fit_params):
        self._set_training(False)
        Xi, yi = batch[self.images_tensor], squeeze(batch[self.labels_tensor])
        with no_grad():
            y_pred = self.infer(Xi, **fit_params)
            loss = self.get_loss(y_pred, yi, X=Xi, training=False)
        return {
            "loss": loss,
            "y_pred": y_pred,
        }


def pytorch_module_to_skorch(
    dataset,
    module,
    criterion,
    device,
    epochs,
    optimizer,
    optimizer_lr,
    dataloader_train_params,
    dataloader_valid_params,
    tensors,
    num_classes
    # skorch_kwargs
):
    images_tensor, labels_tensor = tensors

    if module is None:
        # Set default module.
        module = resnet18()

        # Check if an image tensor is grayscale.
        # TODO: Check if this is correct way to do this.
        if len(dataset[images_tensor].shape) < 4:
            module.conv1 = Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Change the last layer to have num_classes output channels.
        module.fc = Linear(module.fc.in_features, num_classes)

    if criterion is None:
        criterion = CrossEntropyLoss()

    if optimizer is None:
        optimizer = Adam

    model = VisionClassifierNet(
        module=module,
        criterion=criterion,
        device=device,
        max_epochs=epochs,
        optimizer=optimizer,
        optimizer__lr=optimizer_lr,
        train_split=None,
        dataloader_train_params=dataloader_train_params,
        dataloader_valid_params=dataloader_valid_params,
        images_tensor=images_tensor,
        labels_tensor=labels_tensor,
        # **skorch_kwargs
    )

    return model
