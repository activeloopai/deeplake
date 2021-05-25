import os
import numpy as np
import pickle  # TODO: NEVER USE PICKLE

from .chunker import join_chunks

from hub import constants
from hub.util.keys import get_meta_key, get_index_map_key

from hub.core.typing import StorageProvider
from typing import Callable, List, Union


def read_tensor_meta(key: str, storage: StorageProvider):
    return pickle.loads(storage[get_meta_key(key)])


def read_index_map(key: str, storage: StorageProvider):
    return pickle.loads(storage[get_index_map_key(key)])


def read_dataset_meta(storage: StorageProvider):
    return pickle.loads(storage[constants.META_FILENAME])


def key_exists(key: str, storage: StorageProvider):
    meta_key = get_meta_key(key)
    index_map_key = get_index_map_key(key)
    return meta_key in storage or index_map_key in storage


def read_array(
    key: str,
    storage: StorageProvider,
    array_slice: slice = slice(None),
) -> np.ndarray:
    """Read and join chunks into an array from storage.

    Args:
        key (str): Key for where the chunks, index_map, and meta are located in `storage` relative to it's root.
        array_slice (slice): Slice that represents which samples to read. Default = slice representing all samples.
        storage (StorageProvider): StorageProvider for reading the chunks, index_map, and meta.

    Returns:
        np.ndarray: Array containing the sample(s) in the `array_slice` slice.
    """

    meta = read_tensor_meta(key, storage)
    index_map = read_index_map(key, storage)

    # TODO: read samples in parallel
    samples = []
    for index_entry in index_map[array_slice]:
        chunks = []
        for chunk_name in index_entry["chunk_names"]:
            chunk_key = os.path.join(key, "chunks", chunk_name)
            chunk = storage[chunk_key]

            chunks.append(chunk)

        combined_bytes = join_chunks(
            chunks,
            index_entry["start_byte"],
            index_entry["end_byte"],
        )

        out_array = np.frombuffer(combined_bytes, dtype=meta["dtype"])
        samples.append(out_array.reshape(index_entry["shape"]))

    return np.array(samples)
