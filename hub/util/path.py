import os

from hub.core.storage import LocalProvider, S3Provider, MemoryProvider


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
