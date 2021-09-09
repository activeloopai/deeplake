from hub.core.storage.provider import StorageProvider
import posixpath

from hub.constants import (
    CHUNKS_FOLDER,
    DATASET_INFO_FILENAME,
    DATASET_LOCK_FILENAME,
    ENCODED_CHUNK_NAMES_FILENAME,
    ENCODED_CHUNK_NAMES_FOLDER,
    FIRST_COMMIT_ID,
    DATASET_META_FILENAME,
    TENSOR_INFO_FILENAME,
    TENSOR_META_FILENAME,
    TENSOR_VERSION_CHUNK_LIST_FILENAME,
)


def get_chunk_key(key: str, chunk_name: str, commit_id: str) -> str:
    if commit_id == FIRST_COMMIT_ID:
        return posixpath.join(key, CHUNKS_FOLDER, f"{chunk_name}")

    return posixpath.join("versions", commit_id, key, CHUNKS_FOLDER, f"{chunk_name}")


def get_dataset_meta_key(commit_id: str) -> str:
    # dataset meta is always relative to the `StorageProvider`'s root
    if commit_id == FIRST_COMMIT_ID:
        return DATASET_META_FILENAME

    return posixpath.join("versions", commit_id, DATASET_META_FILENAME)


def get_dataset_info_key(commit_id: str) -> str:
    # dataset info is always relative to the `StorageProvider`'s root
    if commit_id == FIRST_COMMIT_ID:
        return DATASET_INFO_FILENAME
    return posixpath.join("versions", commit_id, DATASET_INFO_FILENAME)


def get_dataset_lock_key() -> str:
    return DATASET_LOCK_FILENAME


def get_tensor_meta_key(key: str, commit_id: str) -> str:
    if commit_id == FIRST_COMMIT_ID:
        return posixpath.join(key, TENSOR_META_FILENAME)
    return posixpath.join("versions", commit_id, key, TENSOR_META_FILENAME)


def get_tensor_info_key(key: str, commit_id: str) -> str:
    if commit_id == FIRST_COMMIT_ID:
        return posixpath.join(key, TENSOR_INFO_FILENAME)
    return posixpath.join("versions", commit_id, key, TENSOR_INFO_FILENAME)


def get_tensor_version_chunk_list_key(key: str, commit_id: str) -> str:
    if commit_id == FIRST_COMMIT_ID:
        return posixpath.join(key, TENSOR_VERSION_CHUNK_LIST_FILENAME)
    return posixpath.join(
        "versions", commit_id, key, TENSOR_VERSION_CHUNK_LIST_FILENAME
    )


def get_chunk_id_encoder_key(key: str, commit_id: str) -> str:
    if commit_id == FIRST_COMMIT_ID:
        return posixpath.join(
            key,
            ENCODED_CHUNK_NAMES_FOLDER,
            ENCODED_CHUNK_NAMES_FILENAME,
        )
    return posixpath.join(
        "versions",
        commit_id,
        key,
        ENCODED_CHUNK_NAMES_FOLDER,
        ENCODED_CHUNK_NAMES_FILENAME,
    )


def dataset_exists(storage: StorageProvider) -> bool:
    try:
        storage[get_dataset_meta_key(FIRST_COMMIT_ID)]
        return True
    except KeyError:
        return False


def tensor_exists(key: str, storage: StorageProvider, commit_id: str) -> bool:
    try:
        storage[get_tensor_meta_key(key, commit_id)]
        return True
    except KeyError:
        return False
