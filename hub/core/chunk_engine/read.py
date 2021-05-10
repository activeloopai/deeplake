import os
import pickle
import numpy as np

from hub.core.storage import MemoryProvider

from .generator import unchunk
from .dummy_util import dummy_compression_map
from .meta import get_meta
from .index_map import get_index_map

from typing import Callable, List, Union

# TODO change storage type to StorageProvider
def read_array(
    key: str,
    index: Union[int, slice],
    storage,
) -> np.ndarray:
    # TODO: docstring
    """
    array <- bytes <- decompressor <- chunks <- storage
    """

    if type(index) == int:
        index = slice(index + 1)

    meta = get_meta(key, storage)

    # TODO: don't get entire index_map, instead read entries
    index_map = get_index_map(key, storage)

    compression = dummy_compression_map[meta["compression"]]
    dtype = meta["dtype"]
    length = meta["length"]

    samples = []
    all_same_shape = True
    last_shape = None

    for index_entry in index_map[index]:
        shape = index_entry["shape"]

        # TODO: make this more concise
        if last_shape is not None and last_shape != shape:
            all_same_shape = False

        # TODO: put this in a separate function/class, ideally that caches decompressed chunks
        def decompressed_chunks_generator():
            for chunk_name in index_entry["chunk_names"]:
                chunk_key = os.path.join(key, "chunks", chunk_name)
                raw_chunk = storage[chunk_key]

                if compression.subject == "chunk":
                    if chunk_name in index_entry["incomplete_chunk_names"]:
                        chunk = raw_chunk
                    else:
                        # TODO: different `meta["version"]`s may have different compressor maps
                        chunk = compression.decompress(raw_chunk)
                else:
                    chunk = raw_chunk

                yield chunk

        b = unchunk(
            list(decompressed_chunks_generator()),
            index_entry["start_byte"],
            index_entry["end_byte"],
        )

        a = np.frombuffer(b, dtype=dtype)
        last_shape = shape
        samples.append(a.reshape(shape))

    if all_same_shape:
        return np.array(samples, dtype=dtype)

    return samples
