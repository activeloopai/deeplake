from deeplake.core.dataset import Dataset

from deeplake.constants import MB
from deeplake.core.storage.gcs import GCSProvider
from deeplake.enterprise.util import raise_indra_installation_error  # type: ignore
from deeplake.core.storage import S3Provider
from deeplake.core.storage.indra import IndraProvider
from deeplake.core.storage.azure import AzureProvider
from deeplake.util.remove_cache import get_base_storage
from deeplake.util.exceptions import EmptyTokenException

from deeplake.util.dataset import try_flushing  # type: ignore
import importlib
import jwt

# Load lazy to avoid cycylic import.
INDRA_API = None


def import_indra_api_silent():
    global INDRA_API
    if INDRA_API:
        return INDRA_API
    if not importlib.util.find_spec("indra"):
        return None
    try:
        from indra import api  # type: ignore

        INDRA_API = api
        return api
    except Exception as e:
        return e


def import_indra_api():
    api = import_indra_api_silent()

    if api is None:
        raise_indra_installation_error()  # type: ignore
    elif isinstance(api, Exception):
        raise_indra_installation_error(api)
    else:
        return api


def _get_indra_ds_from_native_provider(provider: IndraProvider):
    api = import_indra_api()
    return api.dataset(provider.core)


def _get_indra_ds_from_azure_provider(
    path: str,
    token: str,
    provider: AzureProvider,
):
    if provider is None:
        return None

    api = import_indra_api()
    account_name = provider.account_name
    account_key = provider.account_key
    sas_token = provider.get_sas_token()
    expiration = str(provider.expiration) if provider.expiration else None

    storage = IndraProvider(
        path,
        read_only=provider.read_only,
        token=token,
        account_name=account_name,
        account_key=account_key,
        sas_token=sas_token,
        expiration=expiration,
    )
    return _get_indra_ds_from_native_provider(storage)


def _get_indra_ds_from_gcp_provider(
    path: str,
    token: str,
    provider: GCSProvider,
):
    if provider is None:
        return None

    api = import_indra_api()
    creds = provider.get_creds()
    creds_dict = {
        k: v
        for k, v in {
            "anon": creds.get("anon", ""),
            "expiration": creds.get("expiration", ""),
            "access_token": creds.get("access_token", ""),
            "json_credentials": creds.get("json_credentials", ""),
            "endpoint_override": creds.get("endpoint_override", ""),
            "scheme": creds.get("scheme", ""),
            "retry_limit_seconds": creds.get("retry_limit_seconds", ""),
            "gcs_oauth_token": creds.get("gcs_oauth_token", ""),
            "project_id": creds.get("project_id", ""),
        }.items()
        if v != ""
    }

    storage = IndraProvider(
        path,
        read_only=provider.read_only,
        token=token,
        origin_path=provider.root,
        **creds_dict,
    )
    return _get_indra_ds_from_native_provider(storage)


def _get_indra_ds_from_s3_provider(
    path: str,
    token: str,
    provider: S3Provider,
):
    if provider is None:
        return None

    api = import_indra_api()

    creds_used = provider.creds_used
    if creds_used == "PLATFORM":
        provider._check_update_creds()
        storage = IndraProvider(
            path,
            read_only=provider.read_only,
            token=token,
            origin_path=provider.root,
            aws_access_key_id=provider.aws_access_key_id,
            aws_secret_access_key=provider.aws_secret_access_key,
            aws_session_token=provider.aws_session_token,
            region_name=provider.aws_region,
            endpoint_url=provider.endpoint_url,
            expiration=str(provider.expiration),
        )
        return _get_indra_ds_from_native_provider(storage)
    elif creds_used == "ENV":
        storage = IndraProvider(
            path,
            read_only=provider.read_only,
            token=token,
            origin_path=provider.root,
            profile_name=provider.profile_name,
        )
        return _get_indra_ds_from_native_provider(storage)
    elif creds_used == "DICT":
        storage = IndraProvider(
            path,
            read_only=provider.read_only,
            token=token,
            origin_path=provider.root,
            profile_name=provider.profile_name,
            aws_access_key_id=provider.aws_access_key_id,
            aws_secret_access_key=provider.aws_secret_access_key,
            aws_session_token=provider.aws_session_token,
            region_name=provider.aws_region,
            endpoint_url=provider.endpoint_url,
        )
        return _get_indra_ds_from_native_provider(storage)


def dataset_to_libdeeplake(hub2_dataset: Dataset):
    """Convert a hub 2.x dataset object to a libdeeplake dataset object."""
    try_flushing(hub2_dataset)
    api = import_indra_api()

    path: str = hub2_dataset.path

    token = (
        hub2_dataset.token
        if hasattr(hub2_dataset, "token") and hub2_dataset.token is not None
        else (
            getattr(hub2_dataset, "_token", None)
            if hasattr(hub2_dataset, "_token") and hub2_dataset._token != ""
            else (
                hub2_dataset.client.get_token()
                if hasattr(hub2_dataset, "client") and hub2_dataset.client
                else ""
            )
        )
    )

    if hub2_dataset.libdeeplake_dataset is not None:
        libdeeplake_dataset = hub2_dataset.libdeeplake_dataset
    elif isinstance(hub2_dataset.storage.next_storage, IndraProvider):
        libdeeplake_dataset = api.dataset(hub2_dataset.storage.next_storage.core)
    else:
        libdeeplake_dataset = None
        if path.startswith("gdrive://"):
            raise ValueError("Gdrive datasets are not supported for libdeeplake")
        elif path.startswith("mem://"):
            raise ValueError("In memory datasets are not supported for libdeeplake")
        elif path.startswith("hub://"):
            provider = hub2_dataset.storage.next_storage
            if isinstance(provider, S3Provider):
                libdeeplake_dataset = _get_indra_ds_from_s3_provider(
                    path=path, token=token, provider=provider
                )

            elif isinstance(provider, GCSProvider):
                libdeeplake_dataset = _get_indra_ds_from_gcp_provider(
                    path=path, token=token, provider=provider
                )

            elif isinstance(provider, AzureProvider):
                libdeeplake_dataset = _get_indra_ds_from_azure_provider(
                    path=path, token=token, provider=provider
                )
            elif isinstance(provider, IndraProvider):
                libdeeplake_dataset = _get_indra_ds_from_native_provider(
                    provider=provider
                )
            else:
                raise ValueError("Unknown storage provider for hub:// dataset")

        elif path.startswith("s3://"):
            libdeeplake_dataset = _get_indra_ds_from_s3_provider(
                path=path, token=token, provider=hub2_dataset.storage.next_storage
            )

        elif path.startswith(("gcs://", "gs://", "gcp://")):
            provider = get_base_storage(hub2_dataset.storage)

            libdeeplake_dataset = _get_indra_ds_from_gcp_provider(
                path=path, token=token, provider=provider
            )

        elif path.startswith(("az://", "azure://")):
            az_provider = get_base_storage(hub2_dataset.storage)
            libdeeplake_dataset = _get_indra_ds_from_azure_provider(
                path=path, token=token, provider=az_provider
            )

        else:
            org_id = hub2_dataset.org_id
            org_id = (
                org_id or jwt.decode(token, options={"verify_signature": False})["id"]
            )
            storage = IndraProvider(
                path, read_only=hub2_dataset.read_only, token=token, org_id=org_id
            )
            libdeeplake_dataset = api.dataset(storage.core)

        hub2_dataset.libdeeplake_dataset = libdeeplake_dataset

    assert libdeeplake_dataset is not None
    try:
        if hasattr(hub2_dataset.storage, "cache_size"):
            libdeeplake_dataset._max_cache_size = max(
                hub2_dataset.storage.cache_size, libdeeplake_dataset._max_cache_size
            )
        commit_id = hub2_dataset.pending_commit_id
        libdeeplake_dataset.checkout(commit_id)
        slice_ = hub2_dataset.index.values[0].value
        if slice_ != slice(None):
            if isinstance(slice_, tuple):
                slice_ = list(slice_)
            libdeeplake_dataset = libdeeplake_dataset[slice_]
    except INDRA_API.api.NoStorageDatasetViewError:
        pass
    return libdeeplake_dataset
