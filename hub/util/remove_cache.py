from hub.core.storage.lru_cache import LRUCache
from hub.core.storage import MemoryProvider
from hub.core.typing import StorageProvider


def remove_memory_cache(storage: StorageProvider):
    if isinstance(storage, LRUCache) and isinstance(
        storage.cache_storage, MemoryProvider
    ):
        return storage.next_storage
    return storage
