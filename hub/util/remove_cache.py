from hub.core.storage.lru_cache import LRUCache
from hub.core.storage import MemoryProvider
from hub.core.typing import StorageProvider


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
        storage = storage.next_storage
    return storage
