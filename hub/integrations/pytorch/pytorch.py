import warnings
from hub.util.shared_memory import get_final_buffer_size
from hub.util.storage import get_pytorch_local_storage
from typing import Callable, Optional, Sequence
from hub.core.storage import SharedMemoryProvider
from hub.core.storage.prefetch_lru_cache import PrefetchLRUCache
from hub.core.storage.shuffle_lru_cache import ShuffleLRUCache
from hub.util.dataset import try_flushing
from hub.util.exceptions import (
    DatasetUnsupportedSharedMemoryCache,
    DatasetUnsupportedPytorch,
    ModuleNotInstalledException,
)
from .pytorch_old import dataset_to_pytorch as old_dataset_to_pytorch
from hub.constants import GB
from hub.util.check_installation import pytorch_installed
from .common import convert_fn as default_convert_fn, collate_fn as default_collate_fn


def set_worker_sharing_strategy(worker_id: int) -> None:
    import torch

    torch.multiprocessing.set_sharing_strategy("file_system")


def dataset_to_pytorch(
    dataset,
    transform: Optional[Callable] = None,
    tensors: Optional[Sequence[str]] = None,
    num_workers: int = 1,
    batch_size: Optional[int] = 1,
    drop_last: bool = False,
    collate_fn: Optional[Callable] = None,
    pin_memory: bool = False,
    shuffle: bool = False,
    buffer_size: int = 10 * 1000,
    use_local_cache: bool = False,
):
    if not pytorch_installed():
        raise ModuleNotInstalledException(
            "'torch' should be installed to convert the Dataset into pytorch format"
        )

    import torch

    try_flushing(dataset)

    class TorchDataset(torch.utils.data.IterableDataset):
        def __init__(
            self,
            dataset,
            transform: Optional[Callable] = None,
            tensors: Optional[Sequence[str]] = None,
            num_workers: int = 1,
            shuffle: bool = False,
            buffer_size_bytes: int = 10 * GB,
            use_local_cache: bool = False,
        ):
            cache = ShuffleLRUCache if shuffle else PrefetchLRUCache
            cache_storage = SharedMemoryProvider()
            next_storage = (
                get_pytorch_local_storage(dataset) if use_local_cache else None
            )

            # currently cache can't work across sessions so it's better to clear it
            if next_storage is not None:
                next_storage.clear()

            try:
                self.cache = cache(
                    cache_storage=cache_storage,
                    next_storage=next_storage,
                    cache_size=buffer_size_bytes,
                    dataset=dataset,
                    num_workers=num_workers,
                    tensor_keys=tensors,
                    transform=transform,
                    mode="pytorch",
                )
            except DatasetUnsupportedSharedMemoryCache:
                raise DatasetUnsupportedPytorch(
                    "Underlying storage of the dataset in MemoryProvider which is not supported."
                )

        def __iter__(self):
            for value in self.cache.iterate_samples():
                if value is not None:
                    yield value

    # TODO new pytorch approach doesn't support 0 workers currently
    num_workers = max(num_workers, 1)
    buffer_size_bytes = get_final_buffer_size(buffer_size)
    if buffer_size_bytes < 1 * GB:
        warnings.warn(
            "The buffer size is currently less than 1000 MB which may lead to issues. "
            "Switching to another pytorch integration which will be slower but more stable within the constraints. "
            "To use the faster integration, consider increasing buffer_size shared memory size or switching to another instance."
        )
        return old_dataset_to_pytorch(
            dataset,
            transform,
            tensors,
            num_workers,
            batch_size,
            drop_last,
            collate_fn,
            pin_memory,
            shuffle,
            buffer_size,
            use_local_cache,
            False,
        )
    pytorch_ds = TorchDataset(
        dataset,
        transform,
        tensors,
        num_workers,
        shuffle,
        buffer_size_bytes,
        use_local_cache,
    )
    if collate_fn is None:
        collate_fn = default_convert_fn if batch_size is None else default_collate_fn
    return torch.utils.data.DataLoader(  # type: ignore
        pytorch_ds,
        batch_size=batch_size,
        drop_last=drop_last,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        worker_init_fn=set_worker_sharing_strategy,
    )
