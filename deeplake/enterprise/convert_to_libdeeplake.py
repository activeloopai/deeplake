from deeplake.core.storage.gcs import GCSProvider
from deeplake.enterprise.util import raise_indra_installation_error  # type: ignore
from deeplake.core.storage import S3Provider
from deeplake.core.storage.azure import AzureProvider
from deeplake.util.remove_cache import get_base_storage
from deeplake.util.exceptions import EmptyTokenException

from deeplake.util.dataset import try_flushing  # type: ignore
import importlib
import jwt

# Load lazy to avoid cycylic import.
INDRA_API = None


def import_indra_api():
    global INDRA_API
    if INDRA_API:
        return INDRA_API
    if not importlib.util.find_spec("indra"):
        raise_indra_installation_error()  # type: ignore
    try:
        from indra import api  # type: ignore

        INDRA_API = api
        return api
    except Exception as e:
        raise_indra_installation_error(e)


INDRA_INSTALLED = bool(importlib.util.find_spec("indra"))


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

    return api.dataset(
        path,
        origin_path=provider.root,
        token=token,
        account_name=account_name,
        account_key=account_key,
        sas_token=sas_token,
        expiration=expiration,
    )


def _get_indra_ds_from_gcp_provider(
    path: str,
    token: str,
    provider: GCSProvider,
):
    if provider is None:
        return None

    api = import_indra_api()
    creds = provider.get_creds()
    anon = creds.get("anon", "")
    expiration = creds.get("expiration", "")
    access_token = creds.get("access_token", "")
    json_credentials = creds.get("json_credentials", "")
    endpoint_override = creds.get("endpoint_override", "")
    scheme = creds.get("scheme", "")
    retry_limit_seconds = creds.get("retry_limit_seconds", "")

    return api.dataset(
        path,
        origin_path=provider.root,
        token=token,
        anon=anon,
        expiration=expiration,
        access_token=access_token,
        json_credentials=json_credentials,
        endpoint_override=endpoint_override,
        scheme=scheme,
        retry_limit_seconds=retry_limit_seconds,
    )


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
        return api.dataset(
            path,
            origin_path=provider.root,
            token=token,
            aws_access_key_id=provider.aws_access_key_id,
            aws_secret_access_key=provider.aws_secret_access_key,
            aws_session_token=provider.aws_session_token,
            region_name=provider.aws_region,
            endpoint_url=provider.endpoint_url,
            expiration=str(provider.expiration),
        )
    elif creds_used == "ENV":
        return api.dataset(
            path,
            origin_path=provider.root,
            token=token,
            profile_name=provider.profile_name,
        )
    elif creds_used == "DICT":
        return api.dataset(
            path,
            origin_path=provider.root,
            token=token,
            aws_access_key_id=provider.aws_access_key_id,
            aws_secret_access_key=provider.aws_secret_access_key,
            aws_session_token=provider.aws_session_token,
            region_name=provider.aws_region,
            endpoint_url=provider.endpoint_url,
        )


def dataset_to_libdeeplake(hub2_dataset):
    """Convert a hub 2.x dataset object to a libdeeplake dataset object."""
    try_flushing(hub2_dataset)
    api = import_indra_api()
    path: str = hub2_dataset.path
    if hub2_dataset.libdeeplake_dataset is None:
        libdeeplake_dataset = None
        if path.startswith("gdrive://"):
            raise ValueError("Gdrive datasets are not supported for libdeeplake")
        elif path.startswith("mem://"):
            raise ValueError("In memory datasets are not supported for libdeeplake")
        elif path.startswith("hub://"):
            token = hub2_dataset._token
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

        elif path.startswith("s3://"):
            s3_provider = hub2_dataset.storage.next_storage
            aws_access_key_id = s3_provider.aws_access_key_id
            aws_secret_access_key = s3_provider.aws_secret_access_key
            aws_session_token = s3_provider.aws_session_token
            region_name = s3_provider.aws_region
            endpoint_url = s3_provider.endpoint_url

            # we don't need to pass profile name as hub has already found creds for it
            libdeeplake_dataset = api.dataset(
                path,
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                aws_session_token=aws_session_token,
                region_name=region_name,
                endpoint_url=endpoint_url,
            )

        elif path.startswith(("gcs://", "gs://", "gcp://")):
            provider = get_base_storage(hub2_dataset.storage)

            libdeeplake_dataset = _get_indra_ds_from_gcp_provider(
                path=path, token=None, provider=provider
            )

        elif path.startswith(("az://", "azure://")):
            az_provider = get_base_storage(hub2_dataset.storage)
            libdeeplake_dataset = _get_indra_ds_from_azure_provider(
                path=path, token=None, provider=az_provider
            )

        else:
            token = hub2_dataset._token
            org_id = hub2_dataset.org_id
            if token is None:
                libdeeplake_dataset = api.dataset(path)
            elif token == "":
                raise EmptyTokenException
            else:
                org_id = (
                    org_id
                    or jwt.decode(token, options={"verify_signature": False})["id"]
                )
                libdeeplake_dataset = api.dataset(path, token=token, org_id=org_id)

        hub2_dataset.libdeeplake_dataset = libdeeplake_dataset
    else:
        libdeeplake_dataset = hub2_dataset.libdeeplake_dataset
    commit_id = hub2_dataset.pending_commit_id
    libdeeplake_dataset.checkout(commit_id)
    slice_ = hub2_dataset.index.values[0].value
    if slice_ != slice(None):
        if isinstance(slice_, tuple):
            slice_ = list(slice_)
        libdeeplake_dataset = libdeeplake_dataset[slice_]
    return libdeeplake_dataset
