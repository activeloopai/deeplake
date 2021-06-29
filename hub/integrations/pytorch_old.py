from typing import Callable, Union, List, Optional, Dict, Tuple
import warnings
from hub.util.exceptions import ModuleNotInstalledException, TensorDoesNotExistError


def dataset_to_pytorch(
    dataset,
    transform: Optional[Callable] = None,
    workers: int = 1,
    tuple_fields: Optional[List[str]] = None,
    python_version_warning: bool = True,
):
    return TorchDataset(
        dataset,
        transform,
        tuple_fields,
        python_version_warning=python_version_warning,
    )


class TorchDataset:
    def __init__(
        self,
        dataset,
        transform: Optional[Callable] = None,
        tuple_fields: Optional[List[str]] = None,
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
        if tuple_fields is None:
            self.keys = list(dataset.tensors)
            self.tuple_mode = False
        else:
            for field in tuple_fields:  # type: ignore
                if field not in dataset.tensors:
                    raise TensorDoesNotExistError(field)
            self.keys = tuple_fields
            self.tuple_mode = True

    def _apply_transform(self, sample: Union[Dict, Tuple]):
        return self.transform(sample) if self.transform else sample

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = {}
        # pytorch doesn't support certain dtypes, which are type casted to another dtype below
        for key in self.keys:
            item = self.dataset[key][index].numpy()
            if item.dtype == "uint16":
                item = item.astype("int32")
            elif item.dtype in ["uint32", "uint64"]:
                item = item.astype("int64")
            sample[key] = item

        if self.tuple_mode:
            sample = tuple(sample[k] for k in self.keys)

        return self._apply_transform(sample)

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]
