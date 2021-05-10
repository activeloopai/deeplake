import os
import pickle
import numpy as np

from hub.core.storage import MemoryProvider

from .generator import unchunk
from .dummy_util import dummy_compression_map
from .meta import get_meta
from .index_map import get_index_map

from hub.core.typing import Provider
from typing import Callable, List, Union, Optional


def read_array(
    key: str,
    storage: Provider,
    index: Optional[Union[int, slice]] = None,
) -> np.ndarray:
    """Read, decompress, & unchunk an array slice from storage.

    Args:
        key (str): Key for where the chunks/index_map/meta will be located in `storage` relative to it's root.
        index (int | slice, optional): Index/slice that represents which samples to read. If `index` is an int value, it
            will be converted into a slice using: `slice(index)`. If no index is provided (default), all samples will be returned.
        storage (Provider): Provider for reading the chunks, index_map, & meta.
    """

    if index is None:
        index = slice(None)
    elif type(index) == int:
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
