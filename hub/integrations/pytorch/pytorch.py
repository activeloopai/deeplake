from hub.util.check_installation import pytorch_installed
from hub.core.storage import PrefetchLRUCache, ShuffleLRUCache, SharedMemoryProvider
from hub.util.dataset import try_flushing
from hub.constants import MB
from typing import Callable, Optional, Sequence
from hub.util.exceptions import ModuleNotInstalledException
from .common import convert_fn as default_convert_fn, collate_fn as default_collate_fn

try:
    import torch
except ModuleNotFoundError:
    pass


def dataset_to_pytorch(
    dataset,
    transform: Optional[Callable] = None,
    tensors: Optional[Sequence[str]] = None,
    num_workers: int = 1,
    batch_size: Optional[int] = 1,
    drop_last: Optional[bool] = False,
    collate_fn: Optional[Callable] = None,
    pin_memory: Optional[bool] = False,
    shuffle: Optional[bool] = False,
):
    if not pytorch_installed:
        raise ModuleNotInstalledException(
            "'torch' should be installed to convert the Dataset into pytorch format"
        )

    try_flushing(dataset)

    class TorchDataset(torch.utils.data.IterableDataset):
        def __init__(
            self,
            dataset,
            transform: Optional[Callable] = None,
            tensors: Optional[Sequence[str]] = None,
            num_workers: int = 1,
            shuffle: bool = False,
        ):
            shm = SharedMemoryProvider()
            size = 10 * 1000 * MB
            if shuffle:
                self.cache = ShuffleLRUCache(
                    shm, None, size, dataset, num_workers, tensors, transform
                )
            else:
                self.cache = PrefetchLRUCache(
                    shm, None, size, dataset, num_workers, tensors, transform
                )

        def __iter__(self):
            yield from self.cache.iterate_samples()

    # TODO new pytorch approach doesn't support 0 workers currently
    num_workers = max(num_workers, 1)
    pytorch_ds = TorchDataset(dataset, transform, tensors, num_workers, shuffle)
    if collate_fn is None:
        collate_fn = default_convert_fn if batch_size is None else default_collate_fn
    return torch.utils.data.DataLoader(  # type: ignore
        pytorch_ds,
        batch_size=batch_size,
        drop_last=drop_last,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )
