import os
import pickle  # TODO: NEVER USE PICKLE
from typing import Callable, List, Optional, Union
import json

import numpy as np
from hub import constants
from hub.core.typing import StorageProvider
from hub.util.keys import get_index_map_key, get_tensor_meta_key


def read_tensor_meta(key: str, storage: StorageProvider) -> dict:
    return json.loads(storage[get_tensor_meta_key(key)])


def read_index_map(key: str, storage: StorageProvider) -> List[dict]:
    return json.loads(storage[get_index_map_key(key)])


def read_dataset_meta(storage: StorageProvider) -> dict:
    return json.loads(storage[constants.META_FILENAME])


def tensor_exists(key: str, storage: StorageProvider) -> bool:
    """A tensor exists if at the specified `key` and `storage` there is both a meta file and index map."""

    meta_key = get_tensor_meta_key(key)
    index_map_key = get_index_map_key(key)
    return meta_key in storage and index_map_key in storage


def read_samples_from_tensor(
    key: str,
    storage: StorageProvider,
    array_slice: slice = slice(None),
) -> np.ndarray:
    """Read (and unchunk) samples from a tensor as an np.ndarray.

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
        array = sample_from_index_entry(key, storage, index_entry, meta["dtype"])
        samples.append(array)

    return np.array(samples)


def sample_from_index_entry(
    key: str, storage: StorageProvider, index_entry: dict, dtype: str
) -> np.ndarray:
    """Get the unchunked sample from a single `index_map` entry."""

    b = bytearray()
    for chunk_name in index_entry["chunk_names"]:
        chunk_key = os.path.join(key, "chunks", chunk_name)
        last_b_len = len(b)
        b.extend(storage[chunk_key])

    start_byte = index_entry["start_byte"]
    end_byte = last_b_len + index_entry["end_byte"]

    return array_from_buffer(
        memoryview(b),
        dtype,
        index_entry["shape"],
        start_byte,
        end_byte,
    )


def array_from_buffer(
    b: memoryview,
    dtype: str,
    shape: tuple = None,
    start_byte: int = 0,
    end_byte: Optional[int] = None,
) -> np.ndarray:
    """Reconstruct a sample from bytes (memoryview) only using the bytes `b[start_byte:end_byte]`. By default all bytes are used."""

    partial_b = b[start_byte:end_byte]
    array = np.frombuffer(partial_b, dtype=dtype)
    if shape is not None:
        array = array.reshape(shape)
    return array
