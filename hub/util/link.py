import warnings
from hub.core.link_creds import LinkCreds
from hub.core.lock import Lock
from hub.core.storage.lru_cache import LRUCache
from hub.util.keys import (
    get_dataset_linked_creds_key,
    get_dataset_linked_creds_lock_key,
)
from hub.util.remove_cache import get_base_storage


def merge_link_creds(old_link_creds: LinkCreds, current_link_creds: LinkCreds):
    num_common_keys = 0
    for key1, key2 in zip(old_link_creds.creds_keys, current_link_creds.creds_keys):
        if key1 == key2:
            num_common_keys += 1
        else:
            break
    new_keys = current_link_creds.creds_keys[num_common_keys:]
    current_link_creds.creds_keys = old_link_creds.creds_keys
    current_link_creds.creds_mapping = old_link_creds.creds_mapping
    current_link_creds.managed_creds_keys = old_link_creds.managed_creds_keys
    current_link_creds.used_creds_keys = old_link_creds.used_creds_keys.union(
        current_link_creds.used_creds_keys
    )
    for key in new_keys:
        if key not in current_link_creds.creds_mapping:
            managed = key in current_link_creds.managed_creds_keys
            current_link_creds.add_creds(key, managed)
    return current_link_creds


def save_link_creds(current_link_creds: LinkCreds, storage: LRUCache):
    """Saves the linked creds info to storage."""
    storage = get_base_storage(storage)
    lock = Lock(storage, get_dataset_linked_creds_lock_key())
    lock.acquire(timeout=10, force=True)
    key = get_dataset_linked_creds_key()
    try:
        data_bytes = storage[key]
    except KeyError:
        data_bytes = None

    if data_bytes is not None:
        old_link_creds = LinkCreds.frombuffer(data_bytes)
        new_link_creds = merge_link_creds(old_link_creds, current_link_creds)
    else:
        new_link_creds = current_link_creds

    storage[key] = new_link_creds.tobytes()
    lock.release()


def warn_missing_managed_creds(self):
    """Warns about any missing managed creds that were added in parallel by someone else."""
    missing_creds = self.link_creds.missing_keys

    missing_managed_creds = [
        creds for creds in missing_creds if creds in self.link_creds.managed_creds_keys
    ]
    if missing_managed_creds:
        warnings.warn(
            f"There are some managed creds missing ({missing_managed_creds}) that were added after the dataset was loaded. Reload the dataset to load them."
        )
