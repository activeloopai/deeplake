import hub
from hub.util.dataset import try_flushing
from hub.core.storage.provider import StorageProvider
from hub.core.storage.lru_cache import LRUCache
from hub.core.storage import MemoryProvider


def remove_memory_cache(storage: StorageProvider):
    """Removes the memory cache."""
    if isinstance(storage, LRUCache) and isinstance(
        storage.cache_storage, MemoryProvider
    ):
        return storage.next_storage
    return storage


def get_base_storage(storage: StorageProvider):
    """Removes all layers of caching and returns the underlying storage."""
    while isinstance(storage, LRUCache):
        if storage.next_storage is not None:
            storage = storage.next_storage
        else:
            storage = storage.cache_storage
    return storage


def get_dataset_with_zero_size_cache(ds):
    """Returns a dataset with same storage but cache size set to zero."""
    try_flushing(ds)
    ds_base_storage = get_base_storage(ds.storage)
    zero_cache_storage = LRUCache(MemoryProvider(), ds_base_storage, 0)
    return hub.Dataset(
        storage=zero_cache_storage,
        index=ds.index,
        group_index=ds.group_index,
        read_only=ds.read_only,
        verbose=False,
    )
