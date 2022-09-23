from hub.core.chunk_engine import ChunkEngine
from hub.core.storage import S3Provider, GCSProvider, GDriveProvider, MemoryProvider
from hub.experimental.util import raise_indra_installation_error  # type: ignore
from hub.util.dataset import try_flushing  # type: ignore
import importlib
import warnings

INDRA_INSTALLED = bool(importlib.util.find_spec("indra"))

if INDRA_INSTALLED:
    try:
        from indra import api  # type:ignore

        INDRA_IMPORT_ERROR = None
    except ImportError as e:
        INDRA_IMPORT_ERROR = e


def dataset_to_hub3(hub2_dataset):
    """Convert a hub 2.x dataset object to a hub 3.x dataset object."""
    raise_indra_installation_error(INDRA_INSTALLED, INDRA_IMPORT_ERROR)
    try_flushing(hub2_dataset)
    path: str = hub2_dataset.path
    if path.startswith("gdrive://"):
        raise ValueError("Gdrive datasets are not supported for hub3")
    elif path.startswith("mem://"):
        raise ValueError("In memory datasets are not supported for hub3")
    elif path.startswith("hub://"):
        token = hub2_dataset._token
        provider = hub2_dataset.storage.next_storage
        if isinstance(provider, S3Provider):
            hub3_dataset = api.dataset(
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
            raise ValueError("GCP datasets are not supported for hub3 currently.")
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
    slice_ = remove_tiled_samples(hub2_dataset, slice_)

    if slice_ != slice(None):
        if isinstance(slice_, tuple):
            slice_ = list(slice_)
        hub3_dataset = hub3_dataset[slice_]
    return hub3_dataset


def verify_base_storage(dataset):
    if isinstance(dataset.base_storage, (GCSProvider, GDriveProvider, MemoryProvider)):
        raise ValueError(
            "GCS, Google Drive and Memory datasets are not supported for experimental features currently."
        )


def remove_tiled_samples(dataset, slice_):
    found_tiled_samples = False
    for tensor in dataset.tensors.values():
        chunk_engine: ChunkEngine = tensor.chunk_engine
        if chunk_engine.tile_encoder_exists:
            tiles = set(chunk_engine.tile_encoder.entries.keys())
            if len(tiles) > 0:
                found_tiled_samples = True
                if isinstance(slice_, slice):
                    start = slice_.start if slice_.start is not None else 0
                    stop = (
                        slice_.stop if slice_.stop is not None else tensor.num_samples
                    )
                    step = slice_.step if slice_.step is not None else 1
                    slice_ = list(range(start, stop, step))
                if isinstance(slice_, (list, tuple)):
                    slice_ = [idx for idx in slice_ if idx not in tiles]

    if found_tiled_samples:
        warnings.warn(
            "One or more tiled samples (big samples that span across multiple chunks) were found in the dataset. These samples are currently not supported for query and dataloader and will be ignored."
        )

    return slice_
