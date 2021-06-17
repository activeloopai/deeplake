from hub.constants import UNCOMPRESSED
from hub.util.compress import decompress_array
import warnings
import os
from typing import Optional
import numpy as np

import numpy as np

from hub.core.typing import StorageProvider


def sample_from_index_entry(
    key: str,
    storage: StorageProvider,
    index_entry: dict,
) -> np.ndarray:
    """Get the un-chunked sample from a single `index_meta` entry.

    Args:
        key (str): Key relative to `storage` where this instance.
        storage (StorageProvider): Storage of the sample.
        index_entry (dict): Index metadata of sample with `chunks_names`, `start_byte` and `end_byte` keys.

    Returns:
        Numpy array from the bytes of the sample.
    """

    chunk_names = index_entry["chunk_names"]
    shape = index_entry["shape"]
    dtype = index_entry["dtype"]

    # sample has no data
    if len(chunk_names) <= 0:
        return np.zeros(shape, dtype=dtype)

    buffer = bytearray()
    for chunk_name in chunk_names:
        chunk_key = os.path.join(key, "chunks", chunk_name)
        last_b_len = len(buffer)
        buffer.extend(storage[chunk_key])

    start_byte = index_entry["start_byte"]
    end_byte = last_b_len + index_entry["end_byte"]

    mv = memoryview(buffer)[start_byte:end_byte]
    sample_compression = index_entry.get("compression", UNCOMPRESSED)

    if sample_compression == UNCOMPRESSED:
        # TODO: chunk-wise compression

        return array_from_buffer(
            mv,
            dtype,
            shape=shape,
        )

    return decompress_array(mv)


def array_from_buffer(
    b: memoryview,
    dtype: str,
    shape: tuple = None,
) -> np.ndarray:
    """Reconstruct a sample from a buffer (memoryview)."""

    array = np.frombuffer(b, dtype=dtype)

    if shape is not None:
        array = array.reshape(shape)
    else:
        warnings.warn(
            "Could not find `shape` for a sample. It was missing from the IndexMeta entry. The array is being returned flat."
        )

    return array
