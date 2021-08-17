from hub.core.storage.shuffle_lru_cache import ShuffleLRUCache
from hub.util.dataset import try_flushing
from hub.constants import MB
from hub.core.storage import SharedMemoryProvider
from typing import Callable, Optional, Sequence
from hub.util.exceptions import ModuleNotInstalledException, TensorDoesNotExistError
from .common import convert_fn as default_convert_fn, collate_fn as default_collate_fn
import torch


def _import_torch():
    global torch
    try:
        import torch
    except ModuleNotFoundError:
        raise ModuleNotInstalledException(
            "'torch' should be installed to convert the Dataset into pytorch format"
        )


def dataset_to_pytorch(
    dataset,
    transform: Optional[Callable] = None,
    tensors: Optional[Sequence[str]] = None,
    num_workers: int = 1,
    batch_size: Optional[int] = 1,
    drop_last: Optional[bool] = False,
    collate_fn: Optional[Callable] = None,
    pin_memory: Optional[bool] = False,
):
    try_flushing(dataset)
    _import_torch()
    # TODO new pytorch approach doesn't support 0 workers currently
    num_workers = max(num_workers, 1)
    pytorch_ds = TorchDataset(dataset, transform, tensors, num_workers)
    if collate_fn is None:
        collate_fn = default_convert_fn if batch_size is None else default_collate_fn
    return torch.utils.data.DataLoader(  # type: ignore
        pytorch_ds,
        batch_size=batch_size,
        drop_last=drop_last,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )


class TorchDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        dataset,
        transform: Optional[Callable] = None,
        tensors: Optional[Sequence[str]] = None,
        num_workers: int = 1,
    ):
        # TODO: shift this to a function
        if tensors is None:
            tensor_keys = list(dataset.tensors)
        else:
            for t in tensors:
                if t not in dataset.tensors:
                    raise TensorDoesNotExistError(t)
            tensor_keys = list(tensors)

        self.cache = ShuffleLRUCache(
            SharedMemoryProvider("abc"),
            None,
            10 * 1000 * MB,
            dataset,
            num_workers,
            tensor_keys,
            transform,
        )

    def __iter__(self):
        yield from self.cache.iterate_samples()
