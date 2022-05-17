import pathlib
from typing import Optional, Union
from hub.core.storage.provider import StorageProvider
import glob
import os


def is_hub_cloud_path(path: str):
    return path.startswith("hub://")


def get_path_from_storage(storage) -> str:
    """Extracts the underlying path from a given storage."""
    from hub.core.storage.lru_cache import LRUCache

    if isinstance(storage, LRUCache):
        return get_path_from_storage(storage.next_storage)
    elif isinstance(storage, StorageProvider):
        if hasattr(storage, "hub_path"):
            return storage.hub_path  # type: ignore
        return storage.root
    else:
        raise ValueError("Invalid storage type.")


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


def get_path_type(path: Optional[str]) -> str:
    if not isinstance(path, str):
        path = str(path)
    if path.startswith("http://") or path.startswith("https://"):
        return "http"
    elif path.startswith("gcs://") or path.startswith("gcp://"):
        return "gcs"
    elif path.startswith("s3://"):
        return "s3"
    elif path.startswith("gdrive://"):
        return "gdrive"
    else:
        return "local"


def is_remote_path(path: str) -> bool:
    return get_path_type(path) != "local"


def convert_pathlib_to_string_if_needed(path: Union[str, pathlib.Path]) -> str:
    if isinstance(path, pathlib.Path):
        path = str(path)
    return path
