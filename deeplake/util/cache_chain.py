from deeplake.constants import LOCAL_CACHE_PREFIX
from typing import List, Optional
from uuid import uuid1
import os
from deeplake.core.storage import (
    StorageProvider,
    MemoryProvider,
    LocalProvider,
)
from deeplake.core.storage.lru_cache import LRUCache
from deeplake.util.exceptions import ProviderSizeListMismatch, ProviderListEmptyError


def get_cache_chain(storage_list: List[StorageProvider], size_list: List[int]):
    """Returns a chain of storage providers as a cache

    Args:
        storage_list (List[StorageProvider]): The list of storage providers needed in a cache.
            Should have atleast one provider in the list.
            If only one provider, LRU cache isn't created and the provider is returned.
        size_list (List[int]): The list of sizes of the caches in bytes.
            Should have size 1 less than provider_list and specifies size of cache for all providers except the last
            one. The last one is the primary storage and is assumed to have infinite space.

    Returns:
        StorageProvider: Returns a cache containing all the storage providers in cache_list if cache_list has 2 or more
            elements.
            Returns the provider if the provider_list has only one provider.

    Raises:
        ProviderListEmptyError: If the provider list is empty.
        ProviderSizeListMismatch: If the len(size_list) + 1 != len(provider_list)
    """
    if not storage_list:
        raise ProviderListEmptyError
    if len(storage_list) <= 1:
        return storage_list[0]
    if len(size_list) + 1 != len(storage_list):
        raise ProviderSizeListMismatch
    store = storage_list[-1]
    for size, cache in zip(reversed(size_list), reversed(storage_list[:-1])):
        store = LRUCache(cache, store, size)
    return store


def generate_chain(
    base_storage: StorageProvider,
    memory_cache_size: int,
    local_cache_size: int,
    path: Optional[str] = None,
) -> StorageProvider:
    """Internal function to be used by Dataset, to generate a cache_chain using a base_storage and sizes of memory and
        local caches.

    Args:
        base_storage (StorageProvider): The underlying actual storage of the Dataset.
        memory_cache_size (int): The size of the memory cache to be used in bytes.
        local_cache_size (int): The size of the local filesystem cache to be used in bytes.
        path (str, optional): The path to the dataset. If not None, it is used to figure out the folder name where the local
            cache is stored.

    Returns:
        StorageProvider: Returns a cache containing the base_storage along with memory cache,
            and local cache if a positive size has been specified for it.
    """

    if path:
        cached_dataset_name = path.replace("://", "_")
        cached_dataset_name = cached_dataset_name.replace("/", "_")
        cached_dataset_name = cached_dataset_name.replace("\\", "_")
    else:
        cached_dataset_name = str(uuid1())

    storage_list: List[StorageProvider] = []
    size_list: List[int] = []

    # Always have a memory cache prefix. Required for support for HubMemoryObjects.
    storage_list.append(MemoryProvider(f"cache/{cached_dataset_name}"))
    size_list.append(memory_cache_size)

    if local_cache_size > 0:
        local_cache_prefix = os.getenv("LOCAL_CACHE_PREFIX", default=LOCAL_CACHE_PREFIX)
        storage_list.append(
            LocalProvider(f"{local_cache_prefix}/{cached_dataset_name}")
        )
        size_list.append(local_cache_size)
    storage_list.append(base_storage)
    return get_cache_chain(storage_list, size_list)
