from hub.util.keys import get_dataset_meta_key, get_tensor_meta_key
from hub.core.storage.provider import StorageProvider
from hub.core.storage import LRUCache
import glob
import os


def get_path_from_storage(storage):
    """Extracts the underlying path from a given storage."""
    if isinstance(storage, LRUCache):
        return get_path_from_storage(storage.next_storage)
    elif isinstance(storage, StorageProvider):
        if hasattr(storage, "hub_path"):
            return storage.hub_path
        return storage.root
    return None


def find_root(path: str) -> str:
    """Find the root of the dataset within the given path.

    Note:
        The "root" is defined as the subdirectory (within path) that has > 1 folder/file (if applicable).
        in other words, if there is a directory structure like:
        dataset -
            Images -
                class1 -
                    img.jpg
                    ...
                class2 -
                    img.jpg
                    ...
                ...

        root is "dataset/Images"

    Args:
        path (str): The local path to folder containing an unstructured dataset and of the form ./path/to/dataset or ~/path/to/dataset or path/to/dataset.

    Returns:
        str: Root path of the unstructured dataset.
    """

    subs = glob.glob(os.path.join(path, "*"))

    subs = [
        sub for sub in subs if os.path.isdir(sub)
    ]  # only keep directories (ignore files)

    if len(subs) == 1:
        return find_root(subs[0])

    return path
