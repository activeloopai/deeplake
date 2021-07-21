from hub.util.keys import get_dataset_meta_key, get_tensor_meta_key
from hub.core.storage.provider import StorageProvider
from hub.core.storage import LRUCache


def get_path_from_storage(storage):
    """Extracts the underlying path from a given storage."""
    if isinstance(storage, LRUCache):
        return get_path_from_storage(storage.next_storage)
    elif isinstance(storage, StorageProvider):
        if hasattr(storage, "hub_path"):
            return storage.hub_path
        return storage.root
    return None
