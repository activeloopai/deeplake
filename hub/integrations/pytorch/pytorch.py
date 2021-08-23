from typing import Callable, Optional, Sequence
from hub.core.storage import SharedMemoryProvider, LocalProvider, LRUCache
from hub.core.storage.prefetch_lru_cache import PrefetchLRUCache
from hub.core.storage.shuffle_lru_cache import ShuffleLRUCache
from hub.util.dataset import try_flushing
from hub.util.check_installation import pytorch_installed
from hub.util.exceptions import (
    DatasetUnsupportedSharedMemoryCache,
    DatasetUnsupportedPytorch,
    ModuleNotInstalledException,
)
from hub.constants import LOCAL_CACHE_PREFIX, SHARED_MEMORY_CACHE_SIZE
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
    drop_last: bool = False,
    collate_fn: Optional[Callable] = None,
    pin_memory: bool = False,
    shuffle: bool = False,
    local_cache_size: int = 0,
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
            local_cache_size: int = 0,
        ):
            cache = ShuffleLRUCache if shuffle else PrefetchLRUCache
            cache_storage = SharedMemoryProvider()
            cache_size = SHARED_MEMORY_CACHE_SIZE
            next_storage = get_next_storage(local_cache_size, dataset)

            try:
                self.cache = cache(
                    cache_storage=cache_storage,
                    next_storage=next_storage,
                    cache_size=cache_size,
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
    pytorch_ds = TorchDataset(
        dataset, transform, tensors, num_workers, shuffle, local_cache_size
    )
    if collate_fn is None:
        collate_fn = default_convert_fn if batch_size is None else default_collate_fn
    return torch.utils.data.DataLoader(  # type: ignore
        pytorch_ds,
        batch_size=batch_size,
        drop_last=drop_last,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )


def get_next_storage(local_cache_size, dataset):
    if local_cache_size > 0:
        local_cache_name: str = dataset.path + "_pytorch"
        local_cache_name = local_cache_name.replace("://", "_")
        local_cache_name = local_cache_name.replace("/", "_")
        local_cache_name = local_cache_name.replace("\\", "_")
        local_cache_path = f"{LOCAL_CACHE_PREFIX}/{local_cache_name}"
        local_provider = LocalProvider(local_cache_path)
        return LRUCache(local_provider, None, local_cache_size)
    else:
        return None
