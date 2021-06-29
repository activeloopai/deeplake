from hub.core.typing import StorageProvider

from hub.util.keys import get_dataset_meta_key


def dataset_exists(storage: StorageProvider) -> bool:
    """A dataset exists if at the specified `storage` there is a dataset meta file."""

    dataset_key = get_dataset_meta_key()
    return dataset_key in storage
