import os
import pickle
import numpy as np

from .write import MemoryProvider

from typing import Callable, List

# TODO change storage type to StorageProvider
def read(
    key: str,
    index: int,
    decompressor: Callable,
    storage: MemoryProvider,
    cache_chain: List[MemoryProvider] = [],
) -> np.ndarray:
    """
    array <- bytes <- decompressor <- chunks <- storage
    """

    # TODO: don't use pickle
    index_map_key = os.path.join(key, "index_map")
    index_map = pickle.loads(storage[index_map_key])

    index_entry = index_map[index]
    # TODO: decode from array instead of dictionary
    start_chunk = index_entry["start_chunk"]
    end_chunk = index_entry["end_chunk"]
    dtype = index_entry["dtype"]
    shape = index_entry["shape"]

    b = bytearray()
    for chunk_index in range(start_chunk, end_chunk + 1):
        # TODO read from caches first
        chunk_key = os.path.join(key, ("c%i" % chunk_index))
        chunk = read_from_storage(chunk_key, storage)
        decompressed_chunk = decompressor(chunk)
        b.extend(decompressed_chunk)

    a = np.frombuffer(b, dtype=dtype)
    return a.reshape(shape)

    return np.array([1])  # TODO


def read_from_cache(key: str, cache_chain: List[MemoryProvider]) -> bool:
    # try to read key from cache, return data if success, else None

    # TODO: cross-cache storage (maybe the data doesn't fit in 1 cache, should we do so partially?)
    for cache_provider in cache_chain:
        try:
            data = cache[key]
            return data
        except:
            pass

    return None


def read_from_storage(key: str, storage: MemoryProvider):
    return storage[key]
