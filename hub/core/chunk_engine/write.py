import numpy as np
from pathlib import Path

from hub.core.chunk_engine import generate_chunks

from typing import Callable


# TODO: remove this (this is just copy & pasted from @Abhinav's dev branch)
class MemoryProvider:
    def __init__(self):
        self.mapper = {}

    def __getitem__(self, path, start_byte=None, end_byte=None):
        return self.mapper[path][slice(start_byte, end_byte)]

    def __setitem__(self, path, value):
        self.mapper[path] = value

    def __iter__(self):
        yield from self.mapper.items()

    def __delitem__(self, path):
        del self.mapper[path]

    def __len__(self):
        return len(self.mapper.keys)


# TODO change storage type to StorageProvider
def write(
    key: str,
    array: np.ndarray,
    storage: MemoryProvider,
    compressor: Callable,
    chunk_size: int,
):
    """
    array -> bytes -> chunks -> compressor -> storage
    """

    path = Path(key)

    b = array.tobytes()
    b = compressor(b)
    last_chunk_num_bytes = None  # TODO
    for i, chunk in enumerate(
        generate_chunks(b, chunk_size, last_chunk_num_bytes=last_chunk_num_bytes)
    ):
        chunk_key = path / ("c%i" % i)
        storage[chunk_key] = b
