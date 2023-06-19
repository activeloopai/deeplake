from deeplake.core.link_creds import LinkCreds
from deeplake.core.lock import Lock
from deeplake.core.storage.lru_cache import LRUCache
from deeplake.util.keys import (
    get_dataset_linked_creds_key,
    get_dataset_linked_creds_lock_key,
)
from deeplake.util.remove_cache import get_base_storage
from typing import Optional, Tuple, List


def merge_link_creds(
    old_link_creds: LinkCreds,
    current_link_creds: LinkCreds,
    replaced_indices: Optional[List[int]] = None,
    managed_info: Optional[Tuple] = None,
):
    num_common_keys = 0
    if replaced_indices is not None:
        new_key = current_link_creds.creds_keys[replaced_indices[0]]
    for i, (key1, key2) in enumerate(
        zip(old_link_creds.creds_keys, current_link_creds.creds_keys)
    ):
        if key1 == key2 or (replaced_indices is not None and i in replaced_indices):
            num_common_keys += 1
        else:
            break
    new_keys = current_link_creds.creds_keys[num_common_keys:]
    current_link_creds.creds_keys = old_link_creds.creds_keys
    current_link_creds.creds_mapping = old_link_creds.creds_mapping
    current_link_creds.managed_creds_keys = old_link_creds.managed_creds_keys.union(
        current_link_creds.managed_creds_keys
    )
    current_link_creds.used_creds_keys = old_link_creds.used_creds_keys.union(
        current_link_creds.used_creds_keys
    )
    for key in new_keys:
        if key not in current_link_creds.creds_mapping:
            managed = key in current_link_creds.managed_creds_keys
            current_link_creds.add_creds_key(key, managed)
    if replaced_indices is not None:
        replaced_key = current_link_creds.creds_keys[replaced_indices[0]]
        current_link_creds.replace_creds(replaced_key, new_key)
    if managed_info is not None:
        is_managed = managed_info[0]
        managed_index = managed_info[1]
        key = current_link_creds.creds_keys[managed_index]
        if is_managed:
            current_link_creds.managed_creds_keys.add(key)
        else:
            current_link_creds.managed_creds_keys.discard(key)
    return current_link_creds


def save_link_creds(
    current_link_creds: LinkCreds,
    storage: LRUCache,
    replaced_indices: Optional[List[int]] = None,
    managed_info: Optional[Tuple] = None,
):
    """Saves the linked creds info to storage."""
    storage = get_base_storage(storage)
    lock = Lock(storage, get_dataset_linked_creds_lock_key())
    lock.acquire(timeout=10)
    key = get_dataset_linked_creds_key()
    try:
        data_bytes = storage[key]
    except KeyError:
        data_bytes = None

    if data_bytes is not None:
        old_link_creds = LinkCreds.frombuffer(data_bytes)
        new_link_creds = merge_link_creds(
            old_link_creds, current_link_creds, replaced_indices, managed_info
        )
    else:
        new_link_creds = current_link_creds

    storage[key] = new_link_creds.tobytes()
    lock.release()


def get_path_creds_key(sample):
    if sample is None:
        return None, None
    return sample.path, sample.creds_key
