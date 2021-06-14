from typing import Callable
import warnings
from hub.util.exceptions import ModuleNotInstalledException


def dataset_to_pytorch(
    dataset, transform: Callable = None, workers: int = 1, python_version_warning=True
):
    return TorchDataset(
        dataset,
        transform,
        python_version_warning=python_version_warning,
    )


class TorchDataset:
    def __init__(
        self,
        dataset,
        transform: Callable = None,
        python_version_warning: bool = True,
    ):
        global torch
        try:
            import torch
        except ModuleNotFoundError:
            raise ModuleNotInstalledException(
                "'torch' should be installed to convert the Dataset into pytorch format"
            )

        if python_version_warning:
            warnings.warn(
                "Python version<3.8 detected. The 'workers' argument will be ignored and Pytorch iteration speeds will be slow. Use newer Python versions for faster Data streaming to Pytorch."
            )

        self.dataset = dataset
        self.transform = transform

    def _apply_transform(self, sample: dict):
        return self.transform(sample) if self.transform else sample

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = {}
        # pytorch doesn't support certain dtypes, which are type casted to another dtype below
        for key in self.dataset.tensors:
            item = self.dataset[key][index].numpy()
            if item.dtype == "uint16":
                item = item.astype("int32")
            elif item.dtype in ["uint32", "uint64"]:
                item = item.astype("int64")
            sample[key] = item
        sample = self._apply_transform(sample)
        return sample

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]
