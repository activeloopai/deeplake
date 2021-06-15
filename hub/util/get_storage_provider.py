from hub.util.exceptions import (
    ImproperDatasetInitialization,
)
from hub.util.tag import check_hub_path
from typing import Optional
from hub.core.storage.provider import StorageProvider
import os
from hub.core.storage import LocalProvider, S3Provider, MemoryProvider
from hub.client.client import HubBackendClient


def get_storage_provider(
    path: Optional[str] = None,
    storage: Optional[StorageProvider] = None,
    mode: Optional[str] = None,
    creds: Optional[dict] = None,
):
    if path is not None and storage is not None:
        raise ImproperDatasetInitialization
    elif path is not None:
        return storage_provider_from_path(path, creds, mode)
    elif storage is not None:
        mode = mode or "a"
        return storage, mode


def storage_provider_from_path(path: str, creds: Optional[dict], mode: Optional[str]):
    """Construct a StorageProvider given a path.

    Arguments:
        path (str): The full path to the Dataset.
        creds (dict): A dictionary containing credentials used to access the dataset at the url.
            This takes precedence over credentials present in the environment. Only used when url is provided. Currently only works with s3 urls.
        mode (str, optional): Mode in which the dataset is opened.
            Supported modes include ("r", "w", "a").

    Returns:
        If given a valid S3 path i.e starts with s3:// returns the S3Provider and mode. (credentials should either be in creds or the environment)
        If given a path starting with mem://return the MemoryProvider and mode.
        If given a valid local path, return the LocalProvider and mode.


    Raises:
        ValueError: If the given path is a local path to a file.
    """
    # TODO pass mode to provider and use it to properly set access.
    mode = mode or "a"
    if creds is None:
        creds = {}
    if path.startswith("s3://"):
        key = creds.get("aws_access_key_id")
        secret = creds.get("aws_secret_access_key")
        token = creds.get("aws_session_token")
        endpoint_url = creds.get("endpoint_url")
        region = creds.get("region")
        return (
            S3Provider(path, key, secret, token, endpoint_url, region, mode=mode),
            mode,
        )
    elif path.startswith("mem://"):
        return MemoryProvider(path), mode
    elif path.startswith("hub://"):
        return storage_provider_from_hub_path(path)
    else:
        if not os.path.exists(path) or os.path.isdir(path):
            return LocalProvider(path), mode
        else:
            raise ValueError(f"Local path {path} must be a path to a local directory")


def storage_provider_from_hub_path(path: str, mode: Optional[str] = None):
    check_hub_path(path)
    tag = path[6:]
    org_id, ds_name = tag.split("/")
    client = HubBackendClient()

    # this will give the proper url (s3, gcs, etc) and corresponding creds, depending on where the dataset is stored.
    url, creds, mode, expiration = client.get_dataset_credentials(org_id, ds_name, mode)
    storage, mode = storage_provider_from_path(url, creds, mode)
    storage._set_hub_creds_info(tag, expiration)
    return storage, mode
