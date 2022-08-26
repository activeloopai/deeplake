import torch
from torchvision.models import resnet18

from skorch import NeuralNet
from skorch.helper import predefined_split

# Wraps the PyTorch Module in an sklearn interface.
class VisionClassifierNet(NeuralNet):
    def __init__(
        self,
        dataloader_train_params,
        dataloader_valid_params,
        images_tensor,
        labels_tensor,
        **kwargs,
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
            print(f"Training on {len(dataset)} examples")
            kwargs = self.dataloader_train_params

        else:
            print(f"Validating on {len(dataset)} examples")
            kwargs = self.dataloader_valid_params
            if kwargs is None:
                kwargs = self.dataloader_train_params
            # Set this to False to avoid getting incorrect probabilities in cross-validation.
            kwargs["shuffle"] = False

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
    dataset,
    dataset_valid,
    module,
    criterion,
    device,
    epochs,
    optimizer,
    optimizer_lr,
    dataloader_train_params,
    dataloader_valid_params,
    tensors,
    num_classes,
):
    images_tensor, labels_tensor = tensors

    if device is None:
        if torch.cuda.is_available():
            device_name = "cuda:0"
        elif torch.backends.mps.is_available():
            device_name = "mps"
        else:
            device_name = "cpu"
        device = torch.device(device_name)

    if module is None:
        # Set default module.
        module = resnet18()

        # Check if an image tensor is grayscale.
        if len(dataset[images_tensor].shape) < 4:
            module.conv1 = torch.nn.Conv2d(
                1, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        # Change the last layer to have num_classes output channels.
        module.fc = torch.nn.Linear(module.fc.in_features, num_classes)

    if criterion is None:
        criterion = torch.nn.CrossEntropyLoss

    if optimizer is None:
        optimizer = torch.optim.Adam

    if dataset_valid:
        model = VisionClassifierNet(
            module=module,
            criterion=criterion,
            device=device,
            max_epochs=epochs,
            optimizer=optimizer,
            optimizer__lr=optimizer_lr,
            train_split=predefined_split(dataset_valid),
            dataloader_train_params=dataloader_train_params,
            dataloader_valid_params=dataloader_valid_params,
            images_tensor=images_tensor,
            labels_tensor=labels_tensor,
        )

    else:
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
        )

    return model
