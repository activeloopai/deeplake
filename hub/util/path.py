from hub.core.storage import LocalProvider
import os


def local_provider_from_path(path: str):
    """Construct a LocalProvider if given a valid local path, or return None

    Arguments:
        path (str): Path that may or may not indicate a local directory

    Returns:
        If the given path could point to a directory, return the LocalProvider.
        If the given path does not look like a local path, return None.

    Raises:
        ValueError: If the given path is a local path to a file
    """
    if path.startswith((".", "/", "~")):
        if not os.path.exists(path) or os.path.isdir(path):
            p = LocalProvider(path)
            return LocalProvider(path)
        else:
            raise ValueError("Local path must be a directory")
    return None
