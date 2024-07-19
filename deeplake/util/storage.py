from deeplake.util.agreement import handle_dataset_agreements
from deeplake.util.cache_chain import generate_chain
from deeplake.constants import LOCAL_CACHE_PREFIX, MB
from deeplake.util.exceptions import AgreementNotAcceptedError
from deeplake.util.tag import process_hub_path
from deeplake.util.path import get_path_type
from typing import Dict, Optional, Union
from deeplake.core.storage.provider import StorageProvider
import os
from deeplake.core.storage import (
    LocalProvider,
    S3Provider,
    GCSProvider,
    AzureProvider,
    MemoryProvider,
    GDriveProvider,
)
from deeplake.client.client import DeepLakeBackendClient
import posixpath
from deeplake.constants import DEFAULT_READONLY
import deeplake.core.dataset


def storage_provider_from_path(
    path: str,
    creds: Optional[Union[dict, str]] = None,
    read_only: bool = False,
    token: Optional[str] = None,
    is_hub_path: bool = False,
    db_engine: bool = False,
    indra: bool = False,
):
    """Construct a StorageProvider given a path.

    Arguments:
        path (str): The full path to the Dataset.
        creds (dict): A dictionary containing credentials used to access the dataset at the url.
            This takes precedence over credentials present in the environment. Only used when url is provided. Currently only works with s3 urls.
        read_only (bool): Opens dataset in read only mode if this is passed as True. Defaults to False.
        token (str): token for authentication into activeloop.
        is_hub_path (bool): Whether the path points to a Deep Lake dataset.
        db_engine (bool): Whether to use Activeloop DB Engine. Only applicable for hub:// paths.
        indra (bool): If true creates indra storage provider.

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

    if indra:
        from deeplake.core.storage.indra import IndraProvider

        storage: StorageProvider = IndraProvider(
            path, read_only=read_only, token=token, creds=creds
        )
    elif path.startswith("hub://"):
        storage = storage_provider_from_hub_path(
            path, read_only, db_engine=db_engine, token=token, creds=creds
        )
    else:
        if path.startswith("s3://"):
            creds_used = "PLATFORM"
            if creds == "ENV":
                creds_used = "ENV"
            elif isinstance(creds, dict) and (
                set(creds.keys()) == {"profile_name"} or not bool(creds)
            ):
                creds_used = "ENV"
            elif isinstance(creds, dict) and bool(creds):
                creds_used = "DICT"
            if isinstance(creds, str):
                creds = {}
            key = creds.get("aws_access_key_id")
            secret = creds.get("aws_secret_access_key")
            session_token = creds.get("aws_session_token")
            endpoint_url = creds.get("endpoint_url")
            region = creds.get("aws_region") or creds.get("region")
            config = creds.get("config", None) or deeplake.config["s3"]
            profile = creds.get("profile_name")
            storage = S3Provider(
                path,
                key,
                secret,
                session_token,
                endpoint_url,
                region,
                profile_name=profile,
                token=token,
                config=config,
            )
            storage.creds_used = creds_used
        else:
            if isinstance(creds, str):
                creds = {}

            if (
                path.startswith("gcp://")
                or path.startswith("gcs://")
                or path.startswith("gs://")
            ):
                storage = GCSProvider(path, creds=creds, token=token)
            elif path.startswith(("az://", "azure://")):
                storage = AzureProvider(path, creds=creds, token=token)
            elif path.startswith("gdrive://"):
                storage = GDriveProvider(path, creds)
            elif path.startswith("mem://"):
                storage = MemoryProvider(path)
            else:
                if not os.path.exists(path) or os.path.isdir(path):
                    storage = LocalProvider(path)
                else:
                    raise ValueError(
                        f"Local path {path} must be a path to a local directory"
                    )
    if not storage._is_hub_path:
        storage._is_hub_path = is_hub_path

    if read_only:
        storage.enable_readonly()
    return storage


def get_dataset_credentials(
    client: DeepLakeBackendClient,
    org_id: str,
    ds_name: str,
    mode: Optional[str],
    db_engine: bool,
):
    # this will give the proper url(s3, gcs, etc) and corresponding creds, depending on where the dataset is stored.
    try:
        url, final_creds, mode, expiration, repo = client.get_dataset_credentials(
            org_id, ds_name, mode=mode, db_engine={"enabled": db_engine}
        )
    except AgreementNotAcceptedError as e:
        handle_dataset_agreements(client, e.agreements, org_id, ds_name)
        url, final_creds, mode, expiration, repo = client.get_dataset_credentials(
            org_id, ds_name, mode=mode, db_engine={"enabled": db_engine}
        )
    return url, final_creds, mode, expiration, repo


def storage_provider_from_hub_path(
    path: str,
    read_only: Optional[bool] = None,
    db_engine: bool = False,
    token: Optional[str] = None,
    creds: Optional[Union[dict, str]] = None,
    indra: bool = False,
):
    path, org_id, ds_name, subdir = process_hub_path(path)
    client = DeepLakeBackendClient(token=token)

    mode = None if (read_only is None) else ("r" if read_only else "w")
    url, final_creds, mode, expiration, repo = get_dataset_credentials(
        client, org_id, ds_name, mode, db_engine
    )

    is_local = get_path_type(url) == "local"

    # ignore mode returned from backend if underlying storage is local
    if mode == "r" and read_only is None and not DEFAULT_READONLY and not is_local:
        # warns user about automatic mode change
        print("Opening dataset in read-only mode as you don't have write permissions.")
        read_only = True

    if read_only is None:
        read_only = DEFAULT_READONLY

    url = posixpath.join(url, subdir)

    creds_used = "PLATFORM"
    if url.startswith("s3://"):
        if creds == "ENV":
            final_creds = {}
            creds_used = "ENV"
        elif isinstance(creds, dict) and set(creds.keys()) == {"profile_name"}:
            final_creds = creds
            creds_used = "ENV"
        elif isinstance(creds, dict) and bool(creds):
            final_creds = creds
            creds_used = "DICT"

    if creds_used != "PLATFORM":
        msg = "Overriding platform credentials with"
        if creds_used == "ENV":
            msg += " credentials loaded from environment."
        elif creds_used == "DICT":
            msg += " credentials from user passed dictionary."
        print(msg)

    storage = storage_provider_from_path(
        path=url,
        creds=final_creds,
        read_only=read_only,
        is_hub_path=True,
        token=token,
        indra=indra,
    )
    storage.creds_used = creds_used
    if creds_used == "PLATFORM":
        storage._set_hub_creds_info(path, expiration, db_engine, repo)

    return storage


def get_storage_and_cache_chain(
    path,
    read_only,
    creds,
    token,
    memory_cache_size,
    local_cache_size,
    db_engine=False,
    indra=False,
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
        db_engine (bool): Whether to use Activeloop DB Engine, only applicable for hub:// paths.
        indra (bool): If true creates indra storage provider.

    Returns:
        A tuple of the storage provider and the storage chain.
    """

    storage = storage_provider_from_path(
        path=path,
        db_engine=db_engine,
        creds=creds,
        read_only=read_only,
        token=token,
        indra=indra,
    )
    memory_cache_size_bytes = memory_cache_size * MB
    local_cache_size_bytes = local_cache_size * MB
    storage_chain = generate_chain(
        storage, memory_cache_size_bytes, local_cache_size_bytes, path
    )
    if storage.read_only:
        storage_chain.enable_readonly()
    return storage, storage_chain


def get_local_storage_path(path: str, prefix: str):
    local_cache_name = path.replace("://", "_")
    local_cache_name = local_cache_name.replace("/", "_")
    local_cache_name = local_cache_name.replace("\\", "_")
    return os.path.join(prefix, local_cache_name)


def get_pytorch_local_storage(dataset: "deeplake.core.dataset.Dataset"):
    """Returns a local storage provider for a given dataset to be used for Pytorch iteration"""
    local_cache_name: str = f"{dataset.path}_pytorch"
    local_cache_prefix = os.getenv("LOCAL_CACHE_PREFIX", default=LOCAL_CACHE_PREFIX)
    local_cache_path = get_local_storage_path(local_cache_name, local_cache_prefix)
    return LocalProvider(local_cache_path)
