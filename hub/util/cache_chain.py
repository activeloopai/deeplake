from hub.core.storage.lru_cache import LRUCache
from typing import List
from hub.core.storage.provider import StorageProvider
from hub.util.exceptions import ProviderSizeListMismatch, ProviderListEmptyError


def get_cache_chain(provider_list: List[StorageProvider], size_list: List[int]):
    """Returns a chain of storage providers as a cache

    Args:
        provider_list (List[StorageProvider]): The list of storage providers needed in a cache.
            Should have atleast one provider in the list.
            If only one provider, LRU cache isn't created and the provider is returned.
        size_list (List[int]): The list of sizes of the caches.
            Should have size 1 less than provider_list and specifies size of cache for all providers except the last one.
            The last one is the primary storage and is assumed to have infinite space.

    Returns:
        StorageProvider: Returns a cache containing all the storage providers in cache_list if cache_list has 2 or more elements.
            Returns the provider if the provider_list has only one provider.

    Raises:
        ProviderListEmptyError: If the provider list is empty.
        ProviderSizeListMismatch: If the len(size_list) + 1 != len(provider_list)
    """
    if not provider_list:
        raise ProviderListEmptyError
    if len(size_list) + 1 != len(provider_list):
        raise ProviderSizeListMismatch
    provider_list.reverse()
    size_list.reverse()
    store = provider_list[0]
    for size, cache in zip(size_list, provider_list[1:]):
        store = LRUCache(cache, store, size)
    return store
