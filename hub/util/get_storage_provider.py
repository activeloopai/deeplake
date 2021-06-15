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
    read_only: bool = False,
    creds: Optional[dict] = None,
):
    if path is not None and storage is not None:
        raise ImproperDatasetInitialization
    elif path is not None:
        return storage_provider_from_path(path, creds, read_only)
    elif storage is not None:
        if read_only:
            storage.enable_readonly()
        return storage


def storage_provider_from_path(
    path: str, creds: Optional[dict], read_only: bool = False
):
    """Construct a StorageProvider given a path.

    Arguments:
        path (str): The full path to the Dataset.
        creds (dict): A dictionary containing credentials used to access the dataset at the url.
            This takes precedence over credentials present in the environment. Only used when url is provided. Currently only works with s3 urls.
        read_only (bool): Opens dataset in read only mode if this is passed as True. Defaults to False.

    Returns:
        If given a valid S3 path i.e starts with s3:// returns the S3Provider and mode. (credentials should either be in creds or the environment)
        If given a path starting with mem://return the MemoryProvider and mode.
        If given a valid local path, return the LocalProvider and mode.


    Raises:
        ValueError: If the given path is a local path to a file.
    """
    if creds is None:
        creds = {}
    if path.startswith("s3://"):
        key = creds.get("aws_access_key_id")
        secret = creds.get("aws_secret_access_key")
        token = creds.get("aws_session_token")
        endpoint_url = creds.get("endpoint_url")
        region = creds.get("region")
        storage: StorageProvider = S3Provider(
            path, key, secret, token, endpoint_url, region
        )
    elif path.startswith("mem://"):
        storage = MemoryProvider(path)
    elif path.startswith("hub://"):
        storage = storage_provider_from_hub_path(path, read_only)
    else:
        if not os.path.exists(path) or os.path.isdir(path):
            storage = LocalProvider(path)
        else:
            raise ValueError(f"Local path {path} must be a path to a local directory")

    if read_only:
        storage.enable_readonly()
    return storage


def storage_provider_from_hub_path(path: str, read_only: bool = False):
    check_hub_path(path)
    tag = path[6:]
    org_id, ds_name = tag.split("/")
    client = HubBackendClient()

    mode = "r" if read_only else None

    # this will give the proper url (s3, gcs, etc) and corresponding creds, depending on where the dataset is stored.
    url, creds, mode, expiration = client.get_dataset_credentials(org_id, ds_name, mode)

    if not read_only and mode == "r":
        # warns user about automatic mode change
        print(
            "Opening Hub Cloud Dataset in read-only mode as you don't have write permissions."
        )
        read_only = True

    storage = storage_provider_from_path(url, creds, read_only)
    storage._set_hub_creds_info(tag, expiration)
    return storage
