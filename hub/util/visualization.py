import numpy as np
from torch.functional import Tensor
from hub.util.exceptions import (
    ModuleNotInstalledException,
    VisualizationError,
)


def _import_plt():
    global plt
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        raise ModuleNotInstalledException(
            "'matplotlib' should be installed to plot data!"
        )


def _imshow(array: np.ndarray):
    if len(array.shape) == 4 and array.shape[0] == 1:
        array = array[0]

    plt.imshow(array)
    plt.show()


def visualize_tensor(tensor: Tensor):
    _import_plt()

    if len(tensor) != 1:
        raise NotImplementedError(
            "Currently only 1 sample can be visualized at a time. Try using `tensor[i].plot()`!"
        )

    try:
        # TODO: support other tensor types here depending on shape, htype, etc.
        _imshow(tensor.numpy())

    except Exception as e:
        reason = "Unknown"
        if hasattr(e, "message"):
            reason = e.message
        raise VisualizationError(tensor.shape, tensor.dtype, reason)
