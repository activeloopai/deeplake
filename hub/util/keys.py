from hub.core.storage.provider import StorageProvider
import posixpath

from hub import constants


def get_chunk_key(key: str, chunk_name: str) -> str:
    return posixpath.join(
        key, constants.CHUNKS_FOLDER, f"{chunk_name}.{constants.CHUNK_EXTENSION}"
    )


def get_dataset_meta_key() -> str:
    # dataset meta is always relative to the `StorageProvider`'s root
    return constants.DATASET_META_FILENAME


def get_tensor_meta_key(key: str) -> str:
    return posixpath.join(key, constants.TENSOR_META_FILENAME)


def get_chunk_id_encoder_key(key: str) -> str:
    return posixpath.join(
        key,
        constants.ENCODED_CHUNK_NAMES_FOLDER,
        constants.ENCODED_CHUNK_NAMES_FILENAME,
    )


def dataset_exists(storage: StorageProvider) -> bool:
    return get_dataset_meta_key() in storage


def tensor_exists(key: str, storage: StorageProvider) -> bool:
    return get_tensor_meta_key(key) in storage
