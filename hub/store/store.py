from hub.store.cache import Cache

from hub.client.hub_control import HubControlClient
import configparser
from typing import Tuple

import fsspec
import gcsfs


def _connect(tag):
    """Connects to the backend and receives credentials"""

    creds = HubControlClient().get_config()
    dataset = HubControlClient().get_dataset_path(tag)

    # If dataset is in DB then return the path
    # Otherwise construct the path from the tag
    if dataset and "path" in dataset:
        path = dataset["path"]
    else:
        sub_tags = tag.split("/")
        # Get repository path from the cred location
        path = "/".join(creds["bucket"].split("/")[:-1])
        path = f"{path}/{sub_tags[0]}/{sub_tags[-1]}"
    return path, creds


def get_fs_and_path(url: str, token=None) -> Tuple[fsspec.AbstractFileSystem, str]:
    if url.startswith("s3://"):
        token = token or dict()
        token = read_aws_creds(token) if isinstance(token, str) else token
        return (
            fsspec.filesystem(
                "s3",
                key=token.get("aws_access_key_id"),
                secret=token.get("aws_secret_access_key"),
                token=token.get("aws_session_token"),
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
        # TOOD check if url is username/dataset:version
        url, creds = _connect(url)
        return (
            fsspec.filesystem(
                "s3",
                key=creds["access_key"],
                secret=creds["secret_key"],
                token=creds["session_token"],
                client_kwargs={
                    "endpoint_url": creds["endpoint"],
                    "region_name": creds["region"],
                },
            ),
            url,
        )


def read_aws_creds(filepath: str):
    parser = configparser.ConfigParser()
    parser.read(filepath)
    return {section: dict(parser.items(section)) for section in parser.sections()}


def _get_storage_map(fs, path):
    return fs.get_mapper(path, check=False, create=False)


def get_storage_map(fs, path, memcache=2 ** 26):
    store = _get_storage_map(fs, path)
    return Cache(store, memcache)
