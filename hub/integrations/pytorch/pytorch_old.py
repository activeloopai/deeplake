import os
import pickle
import warnings
import hub
from typing import Callable, Union, Optional, Dict, Tuple, Sequence
from hub.core.storage import MemoryProvider, LRUCache
from hub.util.dataset import try_flushing
from hub.util.remove_cache import get_base_storage
from hub.util.iterable_ordered_dict import IterableOrderedDict
from hub.util.exceptions import (
    DatasetUnsupportedPytorch,
    ModuleNotInstalledException,
    TensorDoesNotExistError,
    SampleDecompressionError,
)
from hub.constants import MB
from .common import convert_fn as default_convert_fn, collate_fn as default_collate_fn


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
    try_flushing(dataset)

    global torch
    try:
        import torch
    except ModuleNotFoundError:
        raise ModuleNotInstalledException(
            "'torch' should be installed to convert the Dataset into pytorch format"
        )

    pytorch_ds = TorchDataset(
        dataset,
        transform,
        tensors,
        python_version_warning=python_version_warning,
    )

    if collate_fn is None:
        collate_fn = default_convert_fn if batch_size is None else default_collate_fn

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
            if os.name == "nt":
                warnings.warn(
                    f"Windows OS detected. Pytorch iteration speeds are up to 500% faster using linux/macOS along with Python version >= 3.8."
                )
            else:
                warnings.warn(
                    f"Python version < 3.8 detected. Pytorch iteration speeds are up to 500% faster on Python version >= 3.8."
                )

        self.dataset = None
        base_storage = get_base_storage(dataset.storage)
        if isinstance(base_storage, MemoryProvider):
            raise DatasetUnsupportedPytorch(
                "Datasets whose underlying storage is MemoryProvider are not supported for Pytorch iteration."
            )
        self.pickled_storage = pickle.dumps(base_storage)
        self.index = dataset.index
        self.length = len(dataset)
        self.transform = transform
        if tensors is None:
            self.tensor_keys = list(dataset.tensors)
        else:
            for t in tensors:
                if t not in dataset.tensors:
                    raise TensorDoesNotExistError(t)
            self.tensor_keys = list(tensors)
        self._num_bad_samples = 0

    def _apply_transform(self, sample: Union[Dict, Tuple]):
        return self.transform(sample) if self.transform else sample

    def _init_ds(self):
        """
        For each process, dataset should be independently loaded
        """
        if self.dataset is None:
            storage = pickle.loads(self.pickled_storage)

            # creating a new cache for each process
            cache_size = 32 * MB * len(self.tensor_keys)
            cached_storage = LRUCache(MemoryProvider(), storage, cache_size)
            self.dataset = hub.core.dataset.Dataset(
                storage=cached_storage, index=self.index, verbose=False
            )

    def __len__(self):
        return self.length

    def get(self, index: int):
        self._init_ds()
        sample = IterableOrderedDict()
        # pytorch doesn't support certain dtypes, which are type casted to another dtype below
        for key in self.tensor_keys:
            try:
                item = self.dataset[key][index].numpy()  # type: ignore
            except SampleDecompressionError as e:
                warnings.warn(
                    f"Skipping corrupt {self.dataset[key].meta.sample_compression} sample."
                )
                return None
            if item.dtype == "uint16":
                item = item.astype("int32")
            elif item.dtype in ["uint32", "uint64"]:
                item = item.astype("int64")
            sample[key] = item

        return self._apply_transform(sample)

    def __getitem__(self, index: int):
        while True:
            if index + self._num_bad_samples >= self.length:
                raise StopIteration()
            val = self.get(index + self._num_bad_samples)
            if val is None:
                self._num_bad_samples += 1
            else:
                return val

    def __iter__(self):
        for index in range(len(self)):
            val = self[index]
            if val is not None:
                yield self[index]
