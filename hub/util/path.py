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
        return storage.root
    return None


def find_root(path: str) -> str:
    # TODO: update docstring
    # TODO: tests

    """
    Find the root of the dataset within the given path.
    the "root" is defined as being the path to a subdirectory within path that has > 1 folder/file (if applicable).
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
    the output of this function should be "dataset/Images/" as that is the root.
    """

    subs = glob.glob(os.path.join(path, "*"))

    # if ignore_sub:
    #     ignore_sub = os.path.join(path, ignore_sub)
    #     if ignore_sub in subs:
    #         subs.remove(ignore_sub)

    subs = [
        sub for sub in subs if os.path.isdir(sub)
    ]  # only keep directories (ignore files)

    if len(subs) == 1:
        return find_root(subs[0])

    return path
