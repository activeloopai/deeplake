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
    for key in new_keys:
        if key not in current_link_creds.creds_mapping:
            current_link_creds.add_creds(key)
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
