import os

from hub import constants


def get_chunk_key(key: str, chunk_name: str) -> str:
    return os.path.join(key, constants.CHUNKS_FOLDER, chunk_name)


def get_dataset_meta_key() -> str:
    # dataset meta is always relative to the `StorageProvider`'s root
    return constants.DATASET_META_FILENAME


def get_tensor_meta_key(key: str) -> str:
    return os.path.join(key, constants.TENSOR_META_FILENAME)


def get_index_map_key(key: str) -> str:
    return os.path.join(key, constants.INDEX_MAP_FILENAME)
