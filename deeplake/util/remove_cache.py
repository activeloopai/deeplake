from typing import Optional
import hub
from hub.core.storage.provider import StorageProvider
from hub.core.storage.lru_cache import LRUCache
from hub.core.storage import MemoryProvider
from hub.constants import MB


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
    ds_base_storage = get_base_storage(ds.storage)
    zero_cache_storage = LRUCache(MemoryProvider(), ds_base_storage, 0)
    commit_id = ds.pending_commit_id
    ds = hub.core.dataset.dataset_factory(
        path=ds.path,
        storage=zero_cache_storage,
        index=ds.index,
        group_index=ds.group_index,
        read_only=ds.read_only,
        token=ds.token,
        verbose=False,
        link_creds=ds.link_creds,
        pad_tensors=ds._pad_tensors,
    )
    if ds.pending_commit_id != commit_id:
        ds.checkout(commit_id)
    return ds


def create_read_copy_dataset(dataset, commit_id: Optional[str] = None):
    """Creates a read-only copy of the given dataset object, without copying underlying data.

    Args:
        dataset: The Dataset object to copy.
        commit_id: The commit id to checkout the new read-only copy to.

    Returns:
        A new Dataset object in read-only mode.
    """
    base_storage = get_base_storage(dataset.storage)
    if isinstance(base_storage, MemoryProvider):
        new_storage = base_storage.copy()
        new_storage.dict = base_storage.dict
    else:
        new_storage = base_storage.copy()
    storage = LRUCache(MemoryProvider(), new_storage, 256 * MB)
    ds = dataset.__class__(
        storage,
        index=dataset.index,
        group_index=dataset.group_index,
        read_only=True,
        public=dataset.public,
        token=dataset._token,
        verbose=False,
        path=dataset.path,
    )
    if commit_id is not None:
        ds.checkout(commit_id)
    return ds
