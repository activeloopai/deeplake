from deeplake.enterprise.util import raise_indra_installation_error  # type: ignore
from deeplake.core.storage import S3Provider

from deeplake.util.dataset import try_flushing  # type: ignore
import importlib

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


def dataset_to_libdeeplake(hub2_dataset):
    """Convert a hub 2.x dataset object to a libdeeplake dataset object."""
    if hub2_dataset.libdeeplake_dataset is None:
        api = import_indra_api()
        try_flushing(hub2_dataset)
        path: str = hub2_dataset.path
        if path.startswith("gdrive://"):
            raise ValueError("Gdrive datasets are not supported for libdeeplake")
        elif path.startswith("mem://"):
            raise ValueError("In memory datasets are not supported for libdeeplake")
        elif path.startswith("hub://"):
            token = hub2_dataset._token
            provider = hub2_dataset.storage.next_storage
            if isinstance(provider, S3Provider):
                libdeeplake_dataset = api.dataset(
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
            else:
                raise ValueError(
                    "GCP datasets are not supported for libdeeplake currently."
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
            raise ValueError(
                "GCP datasets are not supported for libdeeplake currently."
            )
        else:
            libdeeplake_dataset = api.dataset(path)
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
