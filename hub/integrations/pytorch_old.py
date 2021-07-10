from hub.core.storage.memory import MemoryProvider
from hub.util.remove_cache import get_base_storage
from hub.util.iterable_ordered_dict import IterableOrderedDict
from typing import Callable, Union, List, Optional, Dict, Tuple, Sequence
import warnings
from hub.util.exceptions import (
    DatasetUnsupportedPytorch,
    ModuleNotInstalledException,
    TensorDoesNotExistError,
)
import hub


def _collate_fn(batch):
    import torch

    elem = batch[0]
    if isinstance(elem, IterableOrderedDict):
        return IterableOrderedDict(
            (key, _collate_fn([d[key] for d in batch])) for key in elem.keys()
        )
    return torch.utils.data._utils.collate.default_collate(batch)


def _convert_fn(data):
    import torch

    elem_type = type(data)
    if isinstance(data, IterableOrderedDict):
        return IterableOrderedDict((k, _convert_fn(v)) for k, v in data.items())
    return torch.utils.data._utils.collate.default_convert(data)


def dataset_to_pytorch(
    dataset,
    transform: Optional[Callable] = None,
    tensors: Optional[Sequence[str]] = None,
    num_workers: int = 1,
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

    if collate_fn is None:
        collate_fn = _convert_fn if batch_size is None else _collate_fn

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

        self.dataset = None

        self.storage = get_base_storage(dataset.storage)
        self.index = dataset.index
        if isinstance(self.storage, MemoryProvider):
            raise DatasetUnsupportedPytorch(
                "Datasets whose underlying storage is MemoryProvider are not supported for Pytorch iteration."
            )

        self.transform = transform
        if tensors is None:
            self.tensor_keys = list(dataset.tensors)
        else:
            self.tensor_keys = list(tensors)

    def _apply_transform(self, sample: Union[Dict, Tuple]):
        return self.transform(sample) if self.transform else sample

    def _init_ds(self):
        """
        For each process, dataset should be independently loaded
        """
        if self.dataset is None:
            self.dataset = hub.Dataset(storage=self.storage, index=self.index)

    def __len__(self):
        self._init_ds()
        return len(self.dataset)

    def __getitem__(self, index: int):
        self._init_ds()
        sample = IterableOrderedDict()
        # pytorch doesn't support certain dtypes, which are type casted to another dtype below
        for key in self.tensor_keys:
            item = self.dataset[key][index].numpy()  # type: ignore
            if item.dtype == "uint16":
                item = item.astype("int32")
            elif item.dtype in ["uint32", "uint64"]:
                item = item.astype("int64")
            sample[key] = item

        return self._apply_transform(sample)

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]
