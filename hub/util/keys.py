from hub.util.exceptions import CorruptedMetaError
from hub.core.storage.provider import StorageProvider
import posixpath

from hub import constants


def get_chunk_key(key: str, chunk_name: str) -> str:
    return posixpath.join(key, constants.CHUNKS_FOLDER, f"{chunk_name}")


def get_dataset_meta_key() -> str:
    # dataset meta is always relative to the `StorageProvider`'s root
    return constants.DATASET_META_FILENAME


def get_dataset_info_key() -> str:
    # dataset info is always relative to the `StorageProvider`'s root
    return constants.DATASET_INFO_FILENAME


def get_tensor_meta_key(key: str) -> str:
    return posixpath.join(key, constants.TENSOR_META_FILENAME)


def get_tensor_info_key(key: str) -> str:
    return posixpath.join(key, constants.TENSOR_INFO_FILENAME)


def get_chunk_id_encoder_key(key: str) -> str:
    return posixpath.join(
        key,
        constants.ENCODED_CHUNK_NAMES_FOLDER,
        constants.ENCODED_CHUNK_NAMES_FILENAME,
    )


def dataset_exists(storage: StorageProvider) -> bool:
    """A dataset exists if the provided `storage` contains a `dataset_meta.json`."""

    meta_exists = get_dataset_meta_key() in storage
    return meta_exists


def tensor_exists(key: str, storage: StorageProvider) -> bool:
    return get_tensor_meta_key(key) in storage
