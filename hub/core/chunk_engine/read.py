import os
import pickle
import numpy as np

from .write import MemoryProvider
from .dummy_util import dummy_compression_map

from typing import Callable, List

# TODO change storage type to StorageProvider
# TODO: read with slice
def read(
    key: str,
    storage: MemoryProvider,
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

    compression = dummy_compression_map[meta["compression"]]
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
            chunk_key = os.path.join(key, ("c%i" % chunk_index))

            chunk = storage[chunk_key]

            # TODO: check if chunk is compressed (if it's incomplete this is likely)
            # TODO: different `meta["version"]`s may have different compressor maps
            decompressed_chunk = compression.decompress(chunk)

            b.extend(decompressed_chunk)

        a = np.frombuffer(b, dtype=dtype)
        last_shape = shape
        samples.append(a.reshape(shape))

    if all_same_shape:
        return np.array(samples, dtype=dtype)

    return samples
