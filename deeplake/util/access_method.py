import os
import time
import deeplake
from deeplake.constants import TIMESTAMP_FILENAME, DOWNLOAD_MANAGED_PATH_SUFFIX
from deeplake.util.exceptions import DatasetHandlerError, UnprocessableEntityException
from deeplake.util.storage import get_local_storage_path, storage_provider_from_path
from deeplake.util.connect_dataset import connect_dataset_entry
from deeplake.util.path import get_path_type
from deeplake.util.keys import get_dataset_linked_creds_key
from deeplake.core.link_creds import LinkCreds


def check_access_method(access_method: str, overwrite: bool):
    if access_method not in ["stream", "download", "local"]:
        raise ValueError(
            f"Invalid access method: {access_method}. Must be one of 'stream', 'download', 'local'"
        )
    if access_method in {"download", "local"}:
        if not os.environ.get("DEEPLAKE_DOWNLOAD_PATH"):
            raise ValueError(
                f"DEEPLAKE_DOWNLOAD_PATH environment variable is not set. Cannot use access method '{access_method}'"
            )
        if overwrite:
            raise ValueError(
                "Cannot use access methods download or local with overwrite=True as these methods only interact with local copy of the dataset."
            )


def parse_access_method(access_method: str):
    num_workers = 0
    scheduler = "threaded"
    download = access_method.startswith("download")
    local = access_method.startswith("local")
    if download or local:
        split = access_method.split(":")
        if len(split) == 1:
            split.extend(("threaded", "0"))
        elif len(split) == 2:
            split.append("threaded" if split[1].isnumeric() else "0")
        elif len(split) >= 3:
            num_integers = sum(1 for i in split if i.isnumeric())
            if num_integers != 1 or len(split) > 3:
                raise ValueError(
                    "Invalid access_method format. Expected format is one of the following: {download, download:scheduler, download:num_workers, download:scheduler:num_workers, download:num_workers:scheduler}"
                )

        access_method = "download" if download else "local"
        num_worker_index = 1 if split[1].isnumeric() else 2
        scheduler_index = 3 - num_worker_index
        num_workers = int(split[num_worker_index])
        scheduler = split[scheduler_index]
    return access_method, num_workers, scheduler


def managed_creds_used_in_dataset(path, creds, token):
    managed_creds_used = False
    if get_path_type(path) == "hub":
        # need to connect dataset to backend if managed creds are used in it
        storage = storage_provider_from_path(
            path, creds=creds, read_only=True, token=token
        )
        linked_creds_key = get_dataset_linked_creds_key()
        try:
            data_bytes = storage[linked_creds_key]
        except KeyError:
            data_bytes = None

        if data_bytes:
            link_creds = LinkCreds.frombuffer(data_bytes)
        else:
            link_creds = LinkCreds()

        managed_creds_used = (
            len(link_creds.managed_creds_keys.intersection(link_creds.used_creds_keys))
            > 0
        )
    return managed_creds_used


def connect_dataset_entry_if_needed(
    path, local_path, download, managed_creds_used, token
):
    if managed_creds_used:
        print(
            "Managed credentials are used in the dataset. Connecting local dataset to backend..."
        )
        connect_path = path + DOWNLOAD_MANAGED_PATH_SUFFIX
        if download:
            try:
                connect_dataset_entry(
                    local_path,
                    "S3_KEY",
                    connect_path,
                    token=token,
                    verbose=False,
                    allow_local=True,
                )
            except UnprocessableEntityException:
                # dataset already connected to backend
                pass
        local_path = connect_path
    return local_path


def get_local_dataset(
    access_method,
    path,
    read_only,
    memory_cache_size,
    local_cache_size,
    creds,
    token,
    org_id,
    verbose,
    ds_exists,
    num_workers,
    scheduler,
    reset,
):
    local_path = get_local_storage_path(path, os.environ["DEEPLAKE_DOWNLOAD_PATH"])
    download = access_method == "download" or (
        access_method == "local" and not deeplake.exists(local_path)
    )

    managed_creds_used = managed_creds_used_in_dataset(path, creds, token)

    if download:
        if not ds_exists:
            raise DatasetHandlerError(
                f"Dataset {path} does not exist. Cannot use access method 'download'"
            )
        deeplake.deepcopy(
            path,
            local_path,
            src_creds=creds,
            token=token,
            num_workers=num_workers,
            scheduler=scheduler,
            progressbar=True,
            verbose=False,
            overwrite=True,
        )

    local_path = connect_dataset_entry_if_needed(
        path, local_path, managed_creds_used, download, token
    )

    ds = deeplake.load(
        local_path,
        read_only=read_only,
        verbose=verbose,
        memory_cache_size=memory_cache_size,
        local_cache_size=local_cache_size,
        token=token,
        org_id=org_id,
        reset=reset,
    )
    if download:
        ds.storage.next_storage[TIMESTAMP_FILENAME] = time.ctime().encode("utf-8")
    else:
        timestamp = ds.storage.next_storage[TIMESTAMP_FILENAME].decode("utf-8")
        print(f"** Loaded local copy of dataset. Downloaded on: {timestamp}")
    return ds
