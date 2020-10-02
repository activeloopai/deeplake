import configparser
import posixpath
from typing import Tuple

import fsspec
import gcsfs

from hub.store.cache import Cache

from .exceptions import NotZarrFolderException


def _flatten(list_):
    """
    Helper function to flatten the list
    """
    return [item for sublist in list_ for item in sublist]


def get_fs_and_path(url: str, token=None) -> Tuple[fsspec.AbstractFileSystem, str]:
    if url.startswith("s3://"):
        token = token or dict()
        token = read_aws_creds(token) if isinstance(token, str) else token
        return (
            fsspec.filesystem(
                "s3",
                key=token.get("aws_access_key_id"),
                secret=token.get("aws_secret_access_key"),
            ),
            url[5:],
        )
    elif url.startswith("gcs://"):
        return gcsfs.GCSFileSystem(token=token), url[6:]
    elif url.startswith("abs://"):
        # TODO: Azure
        raise NotImplementedError()
    elif (
        url.startswith("../")
        or url.startswith("./")
        or url.startswith("/")
        or url.startswith("~/")
    ):
        return fsspec.filesystem("file"), url
    else:
        raise NotImplementedError()


def read_aws_creds(filepath: str):
    parser = configparser.ConfigParser()
    parser.read(filepath)
    return {section: dict(parser.items(section)) for section in parser.sections()}


def _get_storage_map(fs, path):
    return fs.get_mapper(path, check=True, create=False)


def get_storage_map(fs, path, memcache=2 ** 26):
    store = _get_storage_map(fs, path)
    return store if memcache == 0 else Cache(store, memcache)
