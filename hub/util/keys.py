from hub.core.storage.provider import StorageProvider
import posixpath

from hub import constants


def get_chunk_key(key: str, chunk_name: str, commit_id: str) -> str:
    if commit_id == "first":
        return posixpath.join(key, constants.CHUNKS_FOLDER, f"{chunk_name}")

    return posixpath.join(
        "versions", commit_id, key, constants.CHUNKS_FOLDER, f"{chunk_name}"
    )


def get_dataset_meta_key(commit_id: str) -> str:
    # dataset meta is always relative to the `StorageProvider`'s root
    if commit_id == "first":
        return constants.DATASET_META_FILENAME

    return posixpath.join("versions", commit_id, constants.DATASET_META_FILENAME)


def get_dataset_info_key(commit_id: str) -> str:
    # dataset info is always relative to the `StorageProvider`'s root
    if commit_id == "first":
        return constants.DATASET_INFO_FILENAME
    return posixpath.join("versions", commit_id, constants.DATASET_INFO_FILENAME)


def get_dataset_lock_key() -> str:
    return constants.DATASET_LOCK_FILENAME


def get_tensor_meta_key(key: str, commit_id: str) -> str:
    if commit_id == "first":
        return posixpath.join(key, constants.TENSOR_META_FILENAME)
    return posixpath.join("versions", commit_id, key, constants.TENSOR_META_FILENAME)


def get_tensor_info_key(key: str, commit_id: str) -> str:
    if commit_id == "first":
        return posixpath.join(key, constants.TENSOR_INFO_FILENAME)
    return posixpath.join("versions", commit_id, key, constants.TENSOR_INFO_FILENAME)


def get_tensor_version_chunk_list_key(key: str, commit_id: str) -> str:
    if commit_id == "first":
        return posixpath.join(key, constants.TENSOR_VERSION_CHUNK_LIST_FILENAME)
    return posixpath.join(
        "versions", commit_id, key, constants.TENSOR_VERSION_CHUNK_LIST_FILENAME
    )


def get_chunk_id_encoder_key(key: str, commit_id: str) -> str:
    if commit_id == "first":
        return posixpath.join(
            key,
            constants.ENCODED_CHUNK_NAMES_FOLDER,
            constants.ENCODED_CHUNK_NAMES_FILENAME,
        )
    return posixpath.join(
        "versions",
        commit_id,
        key,
        constants.ENCODED_CHUNK_NAMES_FOLDER,
        constants.ENCODED_CHUNK_NAMES_FILENAME,
    )


def dataset_exists(storage: StorageProvider) -> bool:
    try:
        storage[get_dataset_meta_key("first")]
        return True
    except KeyError:
        return False


def tensor_exists(key: str, storage: StorageProvider, commit_id: str) -> bool:
    try:
        storage[get_tensor_meta_key(key, commit_id)]
        return True
    except KeyError:
        return False
