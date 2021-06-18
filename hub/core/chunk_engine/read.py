from hub.core.meta.tensor_meta import TensorMeta
from hub.constants import UNCOMPRESSED
from hub.core.compression import decompress_array
import os
import numpy as np

import numpy as np

from hub.core.typing import StorageProvider


def sample_from_index_entry(
    key: str,
    storage: StorageProvider,
    index_entry: dict,
    tensor_meta: TensorMeta,
) -> np.ndarray:
    """Get the un-chunked sample from a single `index_meta` entry.

    Args:
        key (str): Key relative to `storage` where this instance.
        storage (StorageProvider): Storage of the sample.
        index_entry (dict): Index metadata of sample with `chunks_names`, `start_byte` and `end_byte` keys.
        tensor_meta (TensorMeta): TensorMeta object that will be read from.

    Returns:
        Numpy array from the bytes of the sample.
    """

    mv = buffer_from_index_entry(key, storage, index_entry)
    is_empty = len(mv) <= 0

    if is_empty or tensor_meta.sample_compression == UNCOMPRESSED:
        # TODO: chunk-wise compression

        return array_from_buffer(
            mv,
            tensor_meta.dtype,
            shape=index_entry["shape"],
        )

    return decompress_array(mv)


def buffer_from_index_entry(
    key: str, storage: StorageProvider, index_entry: dict
) -> memoryview:
    chunk_names = index_entry["chunk_names"]

    # sample has no data
    if len(chunk_names) <= 0:
        return memoryview(bytes())

    buffer = bytearray()
    for chunk_name in chunk_names:
        chunk_key = os.path.join(key, "chunks", chunk_name)
        last_b_len = len(buffer)
        buffer.extend(storage[chunk_key])

    start_byte = index_entry["start_byte"]
    end_byte = last_b_len + index_entry["end_byte"]

    return memoryview(buffer)[start_byte:end_byte]


def array_from_buffer(
    b: memoryview,
    dtype: str,
    shape: tuple = None,
) -> np.ndarray:
    """Reconstruct a sample from a buffer (memoryview)."""

    array = np.frombuffer(b, dtype=dtype)

    if shape is not None:
        array = array.reshape(shape)

    return array
