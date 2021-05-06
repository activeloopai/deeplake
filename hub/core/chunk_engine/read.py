import os
import pickle
import numpy as np

from .write import MemoryProvider

from typing import Callable, List

# TODO change storage type to StorageProvider
# TODO: read with slice
def read(
    key: str,
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

    # TODO: don't use pickle
    meta_key = os.path.join(key, "meta.json")
    meta = pickle.loads(storage[meta_key])

    dtype = meta["dtype"]
    length = meta["length"]

    samples = []
    all_same_shape = True
    last_shape = None
    for index in range(length):
        index_entry = index_map[index]
        # TODO: decode from array instead of dictionary
        start_chunk = index_entry["start_chunk"]
        end_chunk = index_entry["end_chunk"]
        shape = index_entry["shape"]

        # TODO: make this more concise
        if last_shape is not None and last_shape != shape:
            all_same_shape = False

        b = bytearray()
        for chunk_index in range(start_chunk, end_chunk + 1):
            # TODO read from caches first
            chunk_key = os.path.join(key, ("c%i" % chunk_index))

            decompressed_chunk = read_decompressed_bytes_with_caching(
                chunk_key, cache_chain, storage, decompressor, check_incomplete=True
            )
            b.extend(decompressed_chunk)

        a = np.frombuffer(b, dtype=dtype)
        last_shape = shape
        samples.append(a.reshape(shape))

    if all_same_shape:
        return np.array(samples, dtype=dtype)

    return samples


def read_decompressed_bytes_with_caching(
    key, cache_chain, storage, decompressor, check_incomplete=False
):
    if len(cache_chain) <= 0:
        # TODO: move into exceptions.py
        raise Exception("At least one cache inside of `cache_chain` is required.")

    b = read_and_decompress_from_cache(
        key, cache_chain, decompressor, check_incomplete=check_incomplete
    )

    if b is None:
        b = read_and_decompress_from_storage(
            key, storage, decompressor, check_incomplete=check_incomplete
        )

    return b


def read_and_decompress_from_cache(
    key: str, cache_chain: List[MemoryProvider], decompressor, check_incomplete=False
) -> bool:
    # try to read key from cache, return data if success, else None

    # TODO: move "_incomplete" to util
    incomplete_key = key + "_incomplete"

    # TODO: cross-cache storage (maybe the data doesn't fit in 1 cache, should we do so partially?)
    for cache in cache_chain:
        if key in cache.mapper:
            return decompressor(cache[key])
        if check_incomplete and incomplete_key in cache.mapper:
            # incomplete is not compressed
            return cache[incomplete_key]

    return None


def read_and_decompress_from_storage(
    key: str, storage: MemoryProvider, decompressor, check_incomplete=False
):
    if key in storage.mapper:
        return decompressor(storage[key])

    incomplete_key = key + "_incomplete"
    if check_incomplete and incomplete_key in storage.mapper:
        # incomplete is not compressed
        # TODO: move "_incomplete" to util
        return storage[incomplete_key]

    # TODO: move to exceptions.py
    keys = [key]
    if check_incomplete:
        keys.append(incomplete_key)
    raise Exception("Could not find key(s) `%s` in storage." % str(keys))
