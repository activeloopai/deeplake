from hub.core.storage.memory import MemoryProvider
from hub.util.remove_cache import get_base_storage
from typing import Callable, Union, List, Optional, Dict, Tuple, Sequence
import warnings
from hub.util.exceptions import (
    DatasetUnsupportedPytorch,
    ModuleNotInstalledException,
    TensorDoesNotExistError,
)
from hub.util.subscript_namedtuple import subscript_namedtuple as namedtuple


def dataset_to_pytorch(
    dataset,
    transform: Optional[Callable] = None,
    num_workers: int = 1,
    tensors: Optional[Sequence[str]] = None,
    batch_size: Optional[int] = 1,
    drop_last: Optional[bool] = False,
    collate_fn: Optional[Callable] = None,
    pin_memory: Optional[bool] = False,
    python_version_warning: bool = True,
):
    global torch
    try:
        import torch
    except ModuleNotFoundError:
        raise ModuleNotInstalledException(
            "'torch' should be installed to convert the Dataset into pytorch format"
        )

    dataset.flush()
    pytorch_ds = TorchDataset(
        dataset,
        transform,
        tensors,
        python_version_warning=python_version_warning,
    )
    # TODO add pytorch for num_workers > 1
    if num_workers > 0:
        raise NotImplementedError(
            "Multiproccessed pytorch training is not support for Python version < 3.8. Please set num_workers equal to 0 or upgrade to python 3.8."
        )
    return torch.utils.data.DataLoader(  # type: ignore
        pytorch_ds,
        num_workers=num_workers,
        batch_size=batch_size,
        drop_last=drop_last,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )


class TorchDataset:
    def __init__(
        self,
        dataset,
        transform: Optional[Callable] = None,
        tensors: Optional[Sequence[str]] = None,
        python_version_warning: bool = True,
    ):

        if python_version_warning:
            warnings.warn(
                "Python version<3.8 detected. Pytorch iteration speeds will be slow. Use newer Python versions for faster data streaming to Pytorch."
            )

        self.dataset = dataset

        base_storage = get_base_storage(dataset.storage)
        if isinstance(base_storage, MemoryProvider):
            raise DatasetUnsupportedPytorch(
                "Datasets whose underlying storage is MemoryProvider are not supported for Pytorch iteration."
            )

        self.transform = transform
        self.tensor_keys: List[str]
        if tensors is not None:
            for t in tensors:
                if t not in dataset.tensors:
                    raise TensorDoesNotExistError(t)
            self.tensor_keys = list(tensors)
        else:
            self.tensor_keys = list(dataset.tensors)
        self._return_type = namedtuple("Tensors", self.tensor_keys)

    def _apply_transform(self, sample: Union[Dict, Tuple]):
        return self.transform(sample) if self.transform else sample

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        sample = self._return_type()
        # pytorch doesn't support certain dtypes, which are type casted to another dtype below
        for key in self.tensor_keys:
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
