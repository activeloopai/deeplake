from hub.core.typing import StorageProvider

from hub.util.keys import get_dataset_meta_key


def dataset_exists(storage: StorageProvider) -> bool:
    """A tensor exists if at the specified `key` and `storage` there is both a meta file and index map."""

    dataset_key = get_dataset_meta_key()
    return dataset_key in storage
