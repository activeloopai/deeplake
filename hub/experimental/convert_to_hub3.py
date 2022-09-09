from hub.experimental.util import raise_indra_installation_error  # type: ignore
from hub.util.dataset import try_flushing  # type: ignore

try:
    from indra import api  # type: ignore

    INDRA_INSTALLED = True
except ImportError:
    INDRA_INSTALLED = False


def dataset_to_hub3(hub2_dataset):
    """Convert a hub 2.x dataset object to a hub 3.x dataset object."""
    raise_indra_installation_error(INDRA_INSTALLED)
    try_flushing(hub2_dataset)
    path: str = hub2_dataset.path
    if path.startswith("gdrive://"):
        raise ValueError("Gdrive datasets are not supported for hub3")
    elif path.startswith("mem://"):
        raise ValueError("In memory datasets are not supported for hub3")
    elif path.startswith("hub://"):
        token = hub2_dataset._token
        hub3_dataset = api.dataset(path, token=token)
    elif path.startswith("s3://"):
        s3_provider = hub2_dataset.storage.next_storage
        aws_access_key_id = s3_provider.aws_access_key_id
        aws_secret_access_key = s3_provider.aws_secret_access_key
        aws_session_token = s3_provider.aws_session_token
        region_name = s3_provider.aws_region
        endpoint_url = s3_provider.endpoint_url

        # we don't need to pass profile name as hub has already found creds for it
        hub3_dataset = api.dataset(
            path,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
            region_name=region_name,
            endpoint_url=endpoint_url,
        )

    elif path.startswith(("gcs://", "gs://", "gcp://")):
        raise ValueError("GCP datasets are not supported for hub3 currently.")
    else:
        hub3_dataset = api.dataset(path)

    commit_id = hub2_dataset.pending_commit_id
    hub3_dataset.checkout(commit_id)
    slice_ = hub2_dataset.index.values[0].value

    if slice_ != slice(None):
        hub3_dataset = hub3_dataset[slice_]
    return hub3_dataset
