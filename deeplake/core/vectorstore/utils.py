import deeplake
from deeplake.constants import MB
from deeplake.enterprise.util import raise_indra_installation_error

try:
    from indra import api

    _INDRA_INSTALLED = True
except Exception:
    _INDRA_INSTALLED = False

import numpy as np

from typing import Dict


def check_indra_installation(exec_option, indra_installed):
    if exec_option == "indra" and not indra_installed:
        raise raise_indra_installation_error(indra_import_error=False)


def create_or_load_dataset(
    dataset_path, token, creds, logger, read_only, exec_option, **kwargs
):
    check_indra_installation(exec_option=exec_option, indra_installed=_INDRA_INSTALLED)
    if dataset_exists(dataset_path, token, creds, **kwargs):
        return load_dataset(dataset_path, token, creds, logger, read_only, **kwargs)

    if "overwrite" in kwargs:
        del kwargs["overwrite"]
    return create_dataset(dataset_path, token, **kwargs)


def dataset_exists(dataset_path, token, creds, **kwargs):
    return (
        deeplake.exists(dataset_path, token=token, **creds)
        and "overwrite" not in kwargs
    )


def load_dataset(dataset_path, token, creds, logger, read_only, **kwargs):
    dataset = deeplake.load(dataset_path, token=token, read_only=read_only, **kwargs)

    logger.warning(
        f"Deep Lake Dataset in {dataset_path} already exists, "
        f"loading from the storage"
    )
    return dataset


def create_dataset(dataset_path, token, **kwargs):
    dataset = deeplake.empty(dataset_path, token=token, overwrite=True, **kwargs)

    with dataset:
        dataset.create_tensor(
            "text",
            htype="text",
            create_id_tensor=False,
            create_sample_info_tensor=False,
            create_shape_tensor=False,
            chunk_compression="lz4",
        )
        dataset.create_tensor(
            "metadata",
            htype="json",
            create_id_tensor=False,
            create_sample_info_tensor=False,
            create_shape_tensor=False,
            chunk_compression="lz4",
        )
        dataset.create_tensor(
            "embedding",
            htype="generic",
            dtype=np.float32,
            create_id_tensor=False,
            create_sample_info_tensor=False,
            max_chunk_size=64 * MB,
            create_shape_tensor=True,
        )
        dataset.create_tensor(
            "ids",
            htype="text",
            create_id_tensor=False,
            create_sample_info_tensor=False,
            create_shape_tensor=False,
            chunk_compression="lz4",
        )
    return dataset


def dp_filter(x: dict, filter: Dict[str, str]) -> bool:
    """Filter helper function for Deep Lake"""
    metadata = x["metadata"].data()["value"]
    return all(k in metadata and v == metadata[k] for k, v in filter.items())
