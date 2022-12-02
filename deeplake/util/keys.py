import posixpath
from deeplake.constants import (
    CHUNKS_FOLDER,
    DATASET_DIFF_FILENAME,
    DATASET_INFO_FILENAME,
    DATASET_LOCK_FILENAME,
    ENCODED_CREDS_FOLDER,
    LINKED_CREDS_FILENAME,
    UNSHARDED_ENCODER_FILENAME,
    ENCODED_CHUNK_NAMES_FOLDER,
    ENCODED_SEQUENCE_NAMES_FOLDER,
    ENCODED_TILE_NAMES_FOLDER,
    FIRST_COMMIT_ID,
    DATASET_META_FILENAME,
    TENSOR_INFO_FILENAME,
    TENSOR_META_FILENAME,
    TENSOR_COMMIT_CHUNK_SET_FILENAME,
    TENSOR_COMMIT_DIFF_FILENAME,
    VERSION_CONTROL_INFO_FILENAME,
    VERSION_CONTROL_INFO_FILENAME_OLD,
    VERSION_CONTROL_INFO_LOCK_FILENAME,
    QUERIES_FILENAME,
    QUERIES_LOCK_FILENAME,
)
from deeplake.util.downsample import get_downsample_factor
from deeplake.util.exceptions import (
    S3GetError,
    S3GetAccessError,
    AuthorizationException,
)


def get_chunk_key(key: str, chunk_name: str, commit_id: str) -> str:
    if commit_id == FIRST_COMMIT_ID:
        return "/".join((key, CHUNKS_FOLDER, f"{chunk_name}"))

    return "/".join(("versions", commit_id, key, CHUNKS_FOLDER, f"{chunk_name}"))


def get_dataset_meta_key(commit_id: str) -> str:
    # dataset meta is always relative to the `StorageProvider`'s root
    if commit_id == FIRST_COMMIT_ID:
        return DATASET_META_FILENAME

    return "/".join(("versions", commit_id, DATASET_META_FILENAME))


def get_dataset_info_key(commit_id: str) -> str:
    # dataset info is always relative to the `StorageProvider`'s root
    if commit_id == FIRST_COMMIT_ID:
        return DATASET_INFO_FILENAME
    return "/".join(("versions", commit_id, DATASET_INFO_FILENAME))


def get_dataset_diff_key(commit_id: str) -> str:
    if commit_id == FIRST_COMMIT_ID:
        return DATASET_DIFF_FILENAME
    return "/".join(("versions", commit_id, DATASET_DIFF_FILENAME))


def get_dataset_linked_creds_key() -> str:
    return LINKED_CREDS_FILENAME


def get_dataset_linked_creds_lock_key() -> str:
    return VERSION_CONTROL_INFO_LOCK_FILENAME


def get_version_control_info_key() -> str:
    return VERSION_CONTROL_INFO_FILENAME


def get_version_control_info_key_old() -> str:
    return VERSION_CONTROL_INFO_FILENAME_OLD


def get_version_control_info_lock_key() -> str:
    return VERSION_CONTROL_INFO_LOCK_FILENAME


def get_dataset_lock_key() -> str:
    return DATASET_LOCK_FILENAME


def get_tensor_meta_key(key: str, commit_id: str) -> str:
    if commit_id == FIRST_COMMIT_ID:
        return "/".join((key, TENSOR_META_FILENAME))
    return "/".join(("versions", commit_id, key, TENSOR_META_FILENAME))


def get_tensor_tile_encoder_key(key: str, commit_id: str) -> str:
    if commit_id == FIRST_COMMIT_ID:
        return "/".join((key, ENCODED_TILE_NAMES_FOLDER, UNSHARDED_ENCODER_FILENAME))
    return "/".join(
        (
            "versions",
            commit_id,
            key,
            ENCODED_TILE_NAMES_FOLDER,
            UNSHARDED_ENCODER_FILENAME,
        )
    )


def get_creds_encoder_key(key: str, commit_id: str) -> str:
    if commit_id == FIRST_COMMIT_ID:
        return "/".join((key, ENCODED_CREDS_FOLDER, UNSHARDED_ENCODER_FILENAME))
    return "/".join(
        (
            "versions",
            commit_id,
            key,
            ENCODED_CREDS_FOLDER,
            UNSHARDED_ENCODER_FILENAME,
        )
    )


def get_tensor_info_key(key: str, commit_id: str) -> str:
    if commit_id == FIRST_COMMIT_ID:
        return "/".join([key, TENSOR_INFO_FILENAME])
    return "/".join(("versions", commit_id, key, TENSOR_INFO_FILENAME))


def get_tensor_commit_chunk_set_key(key: str, commit_id: str) -> str:
    if commit_id == FIRST_COMMIT_ID:
        return "/".join((key, TENSOR_COMMIT_CHUNK_SET_FILENAME))
    return "/".join(("versions", commit_id, key, TENSOR_COMMIT_CHUNK_SET_FILENAME))


def get_tensor_commit_diff_key(key: str, commit_id: str) -> str:
    if commit_id == FIRST_COMMIT_ID:
        return "/".join((key, "commit_diff"))
    return "/".join(("versions", commit_id, key, TENSOR_COMMIT_DIFF_FILENAME))


def get_chunk_id_encoder_key(key: str, commit_id: str) -> str:
    if commit_id == FIRST_COMMIT_ID:
        return "/".join(
            (
                key,
                ENCODED_CHUNK_NAMES_FOLDER,
                UNSHARDED_ENCODER_FILENAME,
            )
        )
    return "/".join(
        (
            "versions",
            commit_id,
            key,
            ENCODED_CHUNK_NAMES_FOLDER,
            UNSHARDED_ENCODER_FILENAME,
        )
    )


def get_sequence_encoder_key(key: str, commit_id: str) -> str:
    if commit_id == FIRST_COMMIT_ID:
        return "/".join(
            (
                key,
                ENCODED_SEQUENCE_NAMES_FOLDER,
                UNSHARDED_ENCODER_FILENAME,
            )
        )
    return "/".join(
        (
            "versions",
            commit_id,
            key,
            ENCODED_SEQUENCE_NAMES_FOLDER,
            UNSHARDED_ENCODER_FILENAME,
        )
    )


def dataset_exists(storage) -> bool:
    try:
        storage[get_dataset_meta_key(FIRST_COMMIT_ID)]
        return True
    except S3GetAccessError as err:
        raise AuthorizationException("The dataset storage cannot be accessed") from err
    except (KeyError, S3GetError) as err:
        return False


def tensor_exists(key: str, storage, commit_id: str) -> bool:
    try:
        storage[get_tensor_meta_key(key, commit_id)]
        return True
    except KeyError:
        return False


def get_queries_key() -> str:
    return QUERIES_FILENAME


def get_queries_lock_key() -> str:
    return QUERIES_LOCK_FILENAME


def filter_name(name: str, group_index: str = "") -> str:
    """Filters tensor name and returns full name of the tensor"""
    name = name.strip("/")

    while "//" in name:
        name = name.replace("//", "/")

    name = posixpath.join(group_index, name)
    return name


def get_sample_info_tensor_key(key: str):
    group, key = posixpath.split(key)
    return posixpath.join(group, f"_{key}_info")


def get_sample_id_tensor_key(key: str):
    group, key = posixpath.split(key)
    return posixpath.join(group, f"_{key}_id")


def get_sample_shape_tensor_key(key: str):
    group, key = posixpath.split(key)
    return posixpath.join(group, f"_{key}_shape")


def get_downsampled_tensor_key(key: str, factor: int):
    group, key = posixpath.split(key)
    if key.startswith("_") and "downsampled" in key:
        current_factor = get_downsample_factor(key)
        factor *= current_factor
        ls = key.split("_")
        ls[-1] = str(factor)
        return "_".join(ls)
    return posixpath.join(group, f"_{key}_downsampled_{factor}")
