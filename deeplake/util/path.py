import pathlib
import posixpath
from typing import Optional, Union, Tuple, Dict

from deeplake.core.storage.provider import StorageProvider
from deeplake.util.tag import process_hub_path
from deeplake.constants import HUB_CLOUD_DEV_USERNAME
from deeplake.util.exceptions import InvalidDatasetNameException
import glob
import os
import re

CLOUD_DS_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]*$")
LOCAL_DS_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_ .-]*$")
_relpath_cache: Dict[str, str] = {}


def is_hub_cloud_path(path: str):
    """Whether given ``path`` is a Deep Lake cloud path."""
    return path.startswith("hub://")


def get_path_from_storage(storage) -> str:
    """Extracts the underlying path from a given storage."""
    from deeplake.core.storage.lru_cache import LRUCache

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
    if path.startswith("hub://"):
        return "hub"
    elif path.startswith(("http://", "https://")):
        return "http"
    elif path.startswith(("gcs://", "gcp://", "gs://")):
        return "gcs"
    elif path.startswith("s3://"):
        return "s3"
    elif path.startswith(("az://", "azure://")):
        return "azure"
    elif path.startswith("gdrive://"):
        return "gdrive"
    else:
        return "local"


def is_remote_path(path: str) -> bool:
    return get_path_type(path) != "local"


def convert_string_to_pathlib_if_needed(path, convert_to_pathlib=False):
    converted_path = pathlib.Path(path)
    if convert_to_pathlib and "//" not in path:
        return converted_path
    return path


def convert_pathlib_to_string_if_needed(path: Union[str, pathlib.Path]) -> str:
    if isinstance(path, pathlib.Path):
        path = str(path)
    return path


def process_dataset_path(path: Union[str, pathlib.Path]) -> Tuple[str, Optional[str]]:
    path = str(path).strip()
    dataset_path, _, address = path.partition("@")
    if not address:
        address = None  # type: ignore
    return dataset_path, address


def get_org_id_and_ds_name(path):
    if is_hub_cloud_path(path):
        _, org_id, ds_name, subdir = process_hub_path(path)
        if subdir:
            ds_name += "/" + subdir
    else:
        org_id = HUB_CLOUD_DEV_USERNAME
        ds_name = path.replace("/", "_").replace(".", "")

    return org_id, ds_name


def verify_dataset_name(path):
    path_type = get_path_type(path)
    ds_name = os.path.split(path)[-1]
    match = True
    if path_type == "local":
        match = bool(LOCAL_DS_NAME_PATTERN.match(ds_name))
    elif "/queries/" not in path:
        match = bool(CLOUD_DS_NAME_PATTERN.match(ds_name))
    if not match:
        raise InvalidDatasetNameException(path_type)


def relpath(path, start):
    """
    Wrapper around posixpath.relpath that caches results to avoid performance overhead
    """
    key = path + "::" + start
    if key not in _relpath_cache:
        if len(_relpath_cache) > 1000:
            ### Simple way to keep the cache from growing too large without doing a full LRU cache.
            ### There should not be that many files that we deal with, so likely we will never even hit this.
            _relpath_cache.clear()
        _relpath_cache[key] = posixpath.relpath(path, start)
    return _relpath_cache[key]
