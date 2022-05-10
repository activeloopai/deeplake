import os
import time
import hub
from hub.constants import TIMESTAMP_FILENAME
from hub.util.exceptions import DatasetHandlerError
from hub.util.storage import get_local_storage_path


def check_access_method(access_method: str, overwrite: bool):
    if access_method not in ["stream", "download", "local"]:
        raise ValueError(
            f"Invalid access method: {access_method}. Must be one of 'stream', 'download', 'local'"
        )
    if access_method in {"download", "local"}:
        if not os.environ.get("HUB_DOWNLOAD_PATH"):
            raise ValueError(
                f"HUB_DOWNLOAD_PATH environment variable is not set. Cannot use access method '{access_method}'"
            )
        if overwrite:
            raise ValueError(
                "Cannot use access methods download or local with overwrite=True as these methods only interact with local copy of the dataset."
            )


def get_local_dataset(
    access_method,
    path,
    read_only,
    memory_cache_size,
    local_cache_size,
    creds,
    token,
    verbose,
    ds_exists,
):
    local_path = get_local_storage_path(path, os.environ["HUB_DOWNLOAD_PATH"])
    if access_method == "download":
        if not ds_exists:
            raise DatasetHandlerError(
                f"Dataset {path} does not exist. Cannot use access method 'download'"
            )
        elif hub.exists(local_path):
            raise DatasetHandlerError(
                f"A dataset already exists at the download location {local_path}. To reuse the dataset, use access method 'local'. If you want to download the dataset again, delete the dataset at the download location and try again."
            )
        hub.deepcopy(
            path,
            local_path,
            src_creds=creds,
            src_token=token,
            num_workers=1,
            scheduler="threaded",
            progressbar=True,
            verbose=False,
        )
    elif not hub.exists(local_path):
        raise DatasetHandlerError(
            f"A dataset does not exist at the download location {local_path}. Cannot use access method 'local'. Use access method 'download' to first download the dataset and then use access method 'local' in subsequent runs."
        )

    ds = hub.load(
        local_path,
        read_only=read_only,
        verbose=verbose,
        memory_cache_size=memory_cache_size,
        local_cache_size=local_cache_size,
    )
    if access_method == "download":
        ds.storage.next_storage[TIMESTAMP_FILENAME] = time.ctime().encode("utf-8")
    else:
        timestamp = ds.storage.next_storage[TIMESTAMP_FILENAME].decode("utf-8")
        print(f"** Loaded local copy of dataset. Downloaded on: {timestamp}")
    return ds
