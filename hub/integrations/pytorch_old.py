from typing import Callable, Union, List, Optional, Dict, Tuple
import warnings
from hub.util.exceptions import ModuleNotInstalledException, TensorDoesNotExistError
from collections import OrderedDict


def dataset_to_pytorch(
    dataset,
    transform: Optional[Callable] = None,
    workers: int = 1,
    tensors: Optional[List[str]] = None,
    python_version_warning: bool = True,
):
    return TorchDataset(
        dataset,
        transform,
        tensors,
        python_version_warning=python_version_warning,
    )


class Tensors(OrderedDict):
    def __iter__(self):
        for v in self.values():
            yield v


class TorchDataset:
    def __init__(
        self,
        dataset,
        transform: Optional[Callable] = None,
        tensors: Optional[List[str]] = None,
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
        self.keys: List[str]
        if tensors is not None:
            for t in tensors:
                if t not in dataset.tensors:
                    raise TensorDoesNotExistError(t)
            self.keys = tensors
        else:
            self.keys = list(dataset.tensors)

    def _apply_transform(self, sample: Union[Dict, Tuple]):
        return self.transform(sample) if self.transform else sample

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int) -> Tensors:
        sample = Tensors()
        # pytorch doesn't support certain dtypes, which are type casted to another dtype below
        for key in self.keys:
            item = self.dataset[key][index].numpy()
            if item.dtype == "uint16":
                item = item.astype("int32")
            elif item.dtype in ["uint32", "uint64"]:
                item = item.astype("int64")
            sample[key] = item

        return self._apply_transform(sample)

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]
