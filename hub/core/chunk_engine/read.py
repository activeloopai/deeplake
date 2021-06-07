import os
from typing import List, Optional
import numpy as np

from hub.util.keys import get_index_map_key

import numpy as np

from hub.core.typing import StorageProvider


def sample_from_index_entry(
    key: str, storage: StorageProvider, index_entry: dict, dtype: str
) -> np.ndarray:
    """Get the un-chunked sample from a single `index_map` entry."""

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
    """Reconstruct a sample from bytearray (memoryview) only using the bytes `b[start_byte:end_byte]`. By default all
    bytes are used."""

    partial_b = b[start_byte:end_byte]
    array = np.frombuffer(partial_b, dtype=dtype)
    if shape is not None:
        array = array.reshape(shape)
    return array
