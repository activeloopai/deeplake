from hub.core.storage.gcs import GCSProvider
from hub.util.cache_chain import generate_chain
from hub.constants import LOCAL_CACHE_PREFIX, MB
from hub.util.tag import check_hub_path
from typing import Optional
from hub.core.storage.provider import StorageProvider
import os
from hub.core.storage import LocalProvider, S3Provider, MemoryProvider, LRUCache
from hub.client.client import HubBackendClient


def storage_provider_from_path(
    path: str,
    creds: Optional[dict],
    read_only: bool = False,
    token: Optional[str] = None,
):
    """Construct a StorageProvider given a path.

    Arguments:
        path (str): The full path to the Dataset.
        creds (dict): A dictionary containing credentials used to access the dataset at the url.
            This takes precedence over credentials present in the environment. Only used when url is provided. Currently only works with s3 urls.
        read_only (bool): Opens dataset in read only mode if this is passed as True. Defaults to False.
        token (str): token for authentication into activeloop

    Returns:
        If given a path starting with s3://  returns the S3Provider.
        If given a path starting with gcp:// or gcs:// returns the GCPProvider.
        If given a path starting with mem:// returns the MemoryProvider.
        If given a path starting with hub:// returns the underlying cloud Provider.
        If given a valid local path, returns the LocalProvider.

    Raises:
        ValueError: If the given path is a local path to a file.
    """
    if creds is None:
        creds = {}
    if path.startswith("s3://"):
        key = creds.get("aws_access_key_id")
        secret = creds.get("aws_secret_access_key")
        session_token = creds.get("aws_session_token")
        endpoint_url = creds.get("endpoint_url")
        region = creds.get("region")
        storage: StorageProvider = S3Provider(
            path, key, secret, session_token, endpoint_url, region, token=token
        )
    elif path.startswith("gcp://") or path.startswith("gcs://"):
        storage = GCSProvider(path, creds)
    elif path.startswith("mem://"):
        storage = MemoryProvider(path)
    elif path.startswith("hub://"):
        storage = storage_provider_from_hub_path(path, read_only, token=token)
    else:
        if not os.path.exists(path) or os.path.isdir(path):
            storage = LocalProvider(path)
        else:
            raise ValueError(f"Local path {path} must be a path to a local directory")

    if read_only:
        storage.enable_readonly()
    return storage


def storage_provider_from_hub_path(
    path: str, read_only: bool = False, token: str = None
):
    check_hub_path(path)
    tag = path[6:]
    org_id, ds_name = tag.split("/")
    client = HubBackendClient(token=token)

    mode = "r" if read_only else None

    # this will give the proper url (s3, gcs, etc) and corresponding creds, depending on where the dataset is stored.
    url, creds, mode, expiration = client.get_dataset_credentials(org_id, ds_name, mode)

    if not read_only and mode == "r":
        # warns user about automatic mode change
        print("Opening dataset in read-only mode as you don't have write permissions.")
        read_only = True

    storage = storage_provider_from_path(url, creds, read_only)
    storage._set_hub_creds_info(path, expiration)
    return storage


def get_storage_and_cache_chain(
    path, read_only, creds, token, memory_cache_size, local_cache_size
):
    """
    Returns storage provider and cache chain for a given path, according to arguments passed.

    Args:
        path (str): The full path to the Dataset.
        creds (dict): A dictionary containing credentials used to access the dataset at the url.
            This takes precedence over credentials present in the environment. Only used when url is provided. Currently only works with s3 urls.
        read_only (bool): Opens dataset in read only mode if this is passed as True. Defaults to False.
        token (str): token for authentication into activeloop
        memory_cache_size (int): The size of the in-memory cache to use.
        local_cache_size (int): The size of the local cache to use.

    Returns:
        A tuple of the storage provider and the storage chain.
    """
    storage = storage_provider_from_path(path, creds, read_only, token)
    memory_cache_size_bytes = memory_cache_size * MB
    local_cache_size_bytes = local_cache_size * MB
    storage_chain = generate_chain(
        storage, memory_cache_size_bytes, local_cache_size_bytes, path
    )
    return storage, storage_chain


def get_pytorch_local_storage(dataset):
    """Returns a local storage provider for a given dataset to be used for Pytorch iteration"""
    local_cache_name: str = dataset.path + "_pytorch"
    local_cache_name = local_cache_name.replace("://", "_")
    local_cache_name = local_cache_name.replace("/", "_")
    local_cache_name = local_cache_name.replace("\\", "_")
    local_cache_path = f"{LOCAL_CACHE_PREFIX}/{local_cache_name}"
    return LocalProvider(local_cache_path)
