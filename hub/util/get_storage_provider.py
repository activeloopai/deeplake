from hub.util.exceptions import (
    ImproperDatasetInitialization,
)
from hub.util.tag import check_tag
from typing import Optional
from hub.core.storage.provider import StorageProvider
import os
from hub.core.storage import LocalProvider, S3Provider, MemoryProvider
from hub.client.client import HubBackendClient


def get_storage_provider(
    tag: Optional[str] = None,
    url: Optional[str] = None,
    storage: Optional[StorageProvider] = None,
    mode: Optional[str] = None,
    creds: Optional[dict] = None,
):
    num_storage_args = sum(a is not None for a in (tag, url, storage))
    if num_storage_args != 1:
        raise ImproperDatasetInitialization
    if tag is not None:
        return storage_provider_from_tag(tag, mode)
    elif url is not None:
        return storage_provider_from_url(url, creds, mode)
    elif storage is not None:
        return storage


def storage_provider_from_url(url: str, creds: Optional[dict], mode: Optional[str]):
    """Construct a StorageProvider given a path.

    Arguments:
        url (str): The full path to the Dataset.
        creds (dict): A dictionary containing credentials used to access the dataset at the url.
            This takes precedence over credentials present in the environment. Only used when url is provided. Currently only works with s3 urls.
        mode (str, optional): Mode in which the dataset is opened.
            Supported modes include ("r", "w", "a").

    Returns:
        If given a valid S3 path, return the S3Provider.
        If given return the MemoryProvider.
        If given a valid local path, return the LocalProvider.


    Raises:
        ValueError: If the given path is a local path to a file.
    """
    # TODO pass mode to provider and use it to properly set access.
    mode = mode or "a"
    if creds is None:
        creds = {}
    if url.startswith("s3://"):
        key = creds.get("aws_access_key_id")
        secret = creds.get("aws_secret_access_key")
        token = creds.get("aws_session_token")
        endpoint_url = creds.get("endpoint_url")
        region = creds.get("region")
        return S3Provider(url, key, secret, token, endpoint_url, region, mode=mode)
    elif url.startswith("mem://"):
        return MemoryProvider(url)
    else:
        if not os.path.exists(url) or os.path.isdir(url):
            return LocalProvider(url)
        else:
            raise ValueError(f"Local path {url} must be a path to a local directory")


def storage_provider_from_tag(
    tag: str, mode: Optional[str] = None, public: Optional[bool] = None
):
    check_tag(tag)
    org_id, ds_name = tag.split("/")
    client = HubBackendClient()
    url, creds, mode, expiration = client.get_dataset_credentials(org_id, ds_name, mode)
    storage = storage_provider_from_url(url, creds, mode, public)
    storage._set_hub_creds_info(tag, expiration)
    return storage
