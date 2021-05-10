import os
import numpy as np
import pickle
from uuid import uuid1

from hub.core.chunk_engine import generate_chunks

from hub.core.typing import Provider
from typing import Any, Callable, List, Tuple

from .meta import (
    has_meta,
    get_meta,
    set_meta,
    validate_and_update_meta,
)
from .index_map import has_index_map, get_index_map, set_index_map
from .util import array_to_bytes, normalize_and_batchify_shape


def write_array(
    array: np.ndarray,
    key: str,
    compression,
    chunk_size: int,
    storage: Provider,
    batched: bool = False,
    tobytes: Callable[[np.ndarray], bytes] = array_to_bytes,
):
    """Chunk & write an array to the given storage.

    Args:
        array (np.ndarray): Array to be chunked/written. Batch axis (`array.shape[0]`) is optional, if `array` does have a
            batch dimension, you should pass `batched=True`.
        key (str): Key for where the chunks/index_map/meta will be located in `storage` relative to it's root.
        compression: Compression object that has methods for `compress`, `decompress`, & `subject`. `subject` decides what
            the `compress`/`decompress` methods will be called upon (ie. chunk/sample).
        chunk_size (int): Each individual chunk will be assigned this many bytes maximum.
        storage (Provider): Provider for storing the chunks, index_map, meta, etc.
        batched (bool): If True, the provied `array`'s first axis (`shape[0]`) will be considered it's batch axis.
            If False, a new axis will be created with a size of 1 (`array.shape[0] == 1`). default=False
        tobytes (Callable): Must accept an `np.ndarray` as it's argument & return `bytes`.
    """

    array = normalize_and_batchify_shape(array, batched=batched)

    if has_meta(key, storage) or has_index_map(key, storage):
        # TODO: support appending
        raise NotImplementedError("Appending is not supported yet.")
    _meta = {
        "compression": compression.__name__,
        "chunk_size": chunk_size,
        "dtype": array.dtype.name,
        "length": array.shape[0],
    }
    meta = validate_and_update_meta(key, storage, **_meta)
    index_map = get_index_map(key, storage)

    for i in range(array.shape[0]):
        sample = array[i]
        if compression.subject == "sample":
            # do sample-wise compression
            sample = compression.compress(sample)

        b = tobytes(sample)

        chunk_gen = generate_chunks(b, chunk_size)
        chunk_names = []
        incomplete_chunk_names = []

        for chunk in chunk_gen:
            chunk_name = _create_chunk_name()

            end_byte = len(chunk)  # end byte is based on the uncompressed chunk

            if len(chunk) >= chunk_size:
                if compression.subject == "chunk":
                    # TODO: add threshold for compressing (in case user specifies like 10gb chunk_size)
                    chunk = compression.compress(chunk)
            else:
                incomplete_chunk_names.append(chunk_name)

            chunk_names.append(chunk_name)
            # TODO: make function:
            chunk_key = os.path.join(key, "chunks", chunk_name)
            storage[chunk_key] = chunk

        # TODO: keep track of `sample.shape` over time & add the max_shape:min_shape interval into meta.json for easy queries
        # TODO: encode index_map_entry as array instead of dictionary
        index_map_entry = {
            "chunk_names": chunk_names,
            "incomplete_chunk_names": incomplete_chunk_names,
            "start_byte": 0,
            "end_byte": end_byte,
            "shape": sample.shape,  # shape per sample for dynamic tensors (if strictly fixed-size, store this in meta)
        }
        index_map.append(index_map_entry)

    set_index_map(key, storage, index_map)
    set_meta(key, storage, meta)


def _create_chunk_name() -> str:
    return str(uuid1())
