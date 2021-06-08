import os
import glob

from hub.core.storage import LocalProvider, S3Provider, MemoryProvider


HUB_DIR = "hub"


def storage_provider_from_path(path: str):
    """Construct a StorageProvider given a path.

    Arguments:
        path (str): Path to the provider root, if any.

    Returns:
        If given a valid local path, return the LocalProvider.
        If given a valid S3 path, return the S3Provider.
        Otherwise, return the MemoryProvider.

    Raises:
        ValueError: If the given path is a local path to a file.
    """
    if path.startswith((".", "/", "~")):
        if not os.path.exists(path) or os.path.isdir(path):
            return LocalProvider(path)
        else:
            raise ValueError(f"Local path {path} must be a directory")
    elif path.startswith("s3://"):
        return S3Provider(path)
    else:
        return MemoryProvider(path)



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
    hub_dir = os.path.join(path, HUB_DIR)

    if hub_dir in subs:
        subs.remove(hub_dir)  # ignore the hub directory
    subs = [
        sub for sub in subs if os.path.isdir(sub)
    ]  # only keep directories (ignore files)

    if len(subs) == 1:
        return find_root(subs[0])

    return path