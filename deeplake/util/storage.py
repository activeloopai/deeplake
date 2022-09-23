from hub.core.storage.gcs import GCSProvider
from hub.util.agreement import handle_dataset_agreements
from hub.util.cache_chain import generate_chain
from hub.constants import LOCAL_CACHE_PREFIX, MB
from hub.util.exceptions import AgreementNotAcceptedError
from hub.util.tag import process_hub_path
from typing import Optional
from hub.core.storage.provider import StorageProvider
import os
from hub.core.storage import (
    LocalProvider,
    S3Provider,
    MemoryProvider,
    GDriveProvider,
)
from hub.client.client import HubBackendClient
import posixpath
from hub.constants import DEFAULT_READONLY


def storage_provider_from_path(
    path: str,
    creds: Optional[dict],
    read_only: bool = False,
    token: Optional[str] = None,
    is_hub_path: bool = False,
):
    """Construct a StorageProvider given a path.

    Arguments:
        path (str): The full path to the Dataset.
        creds (dict): A dictionary containing credentials used to access the dataset at the url.
            This takes precedence over credentials present in the environment. Only used when url is provided. Currently only works with s3 urls.
        read_only (bool): Opens dataset in read only mode if this is passed as True. Defaults to False.
        token (str): token for authentication into activeloop.
        is_hub_path (bool): whether the path points to a hub dataset.

    Returns:
        If given a path starting with s3:// returns the S3Provider.
        If given a path starting with gcp:// or gcs:// returns the GCPProvider.
        If given a path starting with gdrive:// returns the GDriveProvider
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
        region = creds.get("aws_region")
        profile = creds.get("profile_name")
        storage: StorageProvider = S3Provider(
            path,
            key,
            secret,
            session_token,
            endpoint_url,
            region,
            profile_name=profile,
            token=token,
        )
    elif (
        path.startswith("gcp://")
        or path.startswith("gcs://")
        or path.startswith("gs://")
    ):
        storage = GCSProvider(path, creds)
    elif path.startswith("gdrive://"):
        storage = GDriveProvider(path, creds)
    elif path.startswith("mem://"):
        storage = MemoryProvider(path)
    elif path.startswith("hub://"):
        storage = storage_provider_from_hub_path(path, read_only, token=token)
    else:
        if not os.path.exists(path) or os.path.isdir(path):
            storage = LocalProvider(path)
        else:
            raise ValueError(f"Local path {path} must be a path to a local directory")
    if not storage._is_hub_path:
        storage._is_hub_path = is_hub_path

    if read_only:
        storage.enable_readonly()
    return storage


def storage_provider_from_hub_path(
    path: str, read_only: bool = False, token: str = None
):
    path, org_id, ds_name, subdir = process_hub_path(path)
    client = HubBackendClient(token=token)

    mode = "r" if read_only else None

    # this will give the proper url (s3, gcs, etc) and corresponding creds, depending on where the dataset is stored.
    try:
        url, creds, mode, expiration = client.get_dataset_credentials(
            org_id, ds_name, mode=mode
        )
    except AgreementNotAcceptedError as e:
        handle_dataset_agreements(client, e.agreements, org_id, ds_name)
        url, creds, mode, expiration = client.get_dataset_credentials(
            org_id, ds_name, mode=mode
        )

    if mode == "r":
        read_only = True
        if read_only is None and not DEFAULT_READONLY:
            # warns user about automatic mode change
            print(
                "Opening dataset in read-only mode as you don't have write permissions."
            )

    if read_only is None:
        read_only = DEFAULT_READONLY

    url = posixpath.join(url, subdir)

    storage = storage_provider_from_path(
        path=url, creds=creds, read_only=read_only, is_hub_path=True
    )
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
    storage = storage_provider_from_path(
        path=path,
        creds=creds,
        read_only=read_only,
        token=token,
    )
    memory_cache_size_bytes = memory_cache_size * MB
    local_cache_size_bytes = local_cache_size * MB
    storage_chain = generate_chain(
        storage, memory_cache_size_bytes, local_cache_size_bytes, path
    )
    return storage, storage_chain


def get_local_storage_path(path: str, prefix: str):
    local_cache_name = path.replace("://", "_")
    local_cache_name = local_cache_name.replace("/", "_")
    local_cache_name = local_cache_name.replace("\\", "_")
    return f"{prefix}/{local_cache_name}"


def get_pytorch_local_storage(dataset):
    """Returns a local storage provider for a given dataset to be used for Pytorch iteration"""
    local_cache_name: str = f"{dataset.path}_pytorch"
    local_cache_path = get_local_storage_path(local_cache_name, LOCAL_CACHE_PREFIX)
    return LocalProvider(local_cache_path)
