from skorch import NeuralNet
from sklearn.metrics import accuracy_score

import torch

# Wraps the PyTorch Module in a scikit-learn interface.
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

    def score(self, X, y):
        pred_probs = self.predict(X)
        y_pred = pred_probs.argmax(-1)
        return accuracy_score(y_pred=y_pred, y_true=y)
