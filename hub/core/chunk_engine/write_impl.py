import os
import numpy as np
import pickle

from hub.core.chunk_engine import generate_chunks

from typing import Callable, Optional, List


# TODO: remove this after abhinav's providers are merged to release/2.0 (this is just copy & pasted from @Abhinav's dev branch)
class MemoryProvider:
    def __init__(self):
        self.mapper = {}
        self.max_bytes = 4096  # TODO

    def __getitem__(self, path, start_byte=None, end_byte=None):
        return self.mapper[path][slice(start_byte, end_byte)]

    def __setitem__(self, path, value):
        self.mapper[path] = value

    def __iter__(self):
        yield from self.mapper.items()

    def __delitem__(self, path):
        del self.mapper[path]

    def __len__(self):
        return len(self.mapper.keys())

    @property
    def used_space(self):
        # TODO: this is a slow operation
        return sum([len(b) for b in self.mapper.values()])

    def has_space(self, num_bytes: int) -> bool:
        space_left = self.max_bytes - self.used_space
        return num_bytes <= space_left


# TODO change storage type to StorageProvider
def write(
    key: str,
    array: np.ndarray,
    compressor: Callable,
    chunk_size: int,
    storage: MemoryProvider,
    cache_chain: List[MemoryProvider] = [],
    batched: bool = False,
):
    """
    array -> bytes -> chunks -> compressor -> storage
    """

    if batched:
        raise NotImplemented

    # TODO: normalize array shape
    # TODO: make sure the provided shape has the same dimensionality of the other samples in the tensor being written to

    index_map = {}  # TODO
    sample_index = 0  # TODO: determine sample index from index_map

    # TODO: hilbert curves? tobytes() doesn't support efficient slicing
    b = array.tobytes()

    last_chunk_num_bytes = None  # TODO
    for chunk_index, chunk in enumerate(
        generate_chunks(b, chunk_size, last_chunk_num_bytes=last_chunk_num_bytes)
    ):
        chunk_key = os.path.join(key, ("c%i" % chunk_index))
        # TODO: don't compress an incomplete chunk (if it isn't == chunk_size it is incomplete)
        compressed_chunk = compressor(chunk)

        if len(cache_chain) <= 0:
            # if `cache_chain` is empty, store to main provider.
            write_to_storage(chunk_key, compressed_chunk, storage)

        else:
            # if `cache_chain` is not empty, prioritize cache storage over main provider.
            cache_success = write_to_cache(chunk_key, compressed_chunk, cache_chain)

            if not cache_success:
                flush_cache(cache_chain, storage)
                cache_success = cache(chunk_key, compressed_chunk, cache_chain)

                if not cache_success:
                    # TODO move into exceptions.py
                    raise Exception("Caching chunk failed even after flushing.")

    # TODO: encode in array instead of dict
    # TODO: start & end bytes
    # TODO: chunk index map
    index_map[sample_index] = {
        "start_chunk": 0,
        "end_chunk": chunk_index,
        "dtype": array.dtype,
        "shape": array.shape,
    }

    # TODO: don't use pickle
    index_map_key = os.path.join(key, "index_map")
    write_to_storage(index_map_key, pickle.dumps(index_map), storage)

    flush_cache(cache_chain, storage)


def write_to_cache(key: str, data: bytes, cache_chain: List[MemoryProvider]) -> bool:
    # max out cache

    # TODO: cross-cache storage (maybe the data doesn't fit in 1 cache, should we do so partially?)
    for cache_provider in cache_chain:
        if cache_provider.has_space(len(data)):
            cache_provider[key] = data
            return True

    return False


def write_to_storage(key: str, data: bytes, storage: MemoryProvider):
    storage[key] = data


def flush_cache(cache_chain: List[MemoryProvider], storage: MemoryProvider):
    # TODO: send all cached data -> storage & clear the caches.

    for cache in cache_chain:
        keys = []
        for key, chunk in cache:
            storage[key] = chunk
            keys.append(key)

        for key in keys:
            del cache[key]

        # TODO: test flushing to make surec cache.used_space will return 0
