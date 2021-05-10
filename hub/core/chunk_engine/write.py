import os
import numpy as np
import pickle

from typing import Any, Callable, List, Tuple

from hub.core.chunk_engine import generate_chunks

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
    storage,
    batched: bool = False,
):
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

    local_chunk_index = 0

    for i in range(array.shape[0]):
        sample = array[i]
        if compression.subject == "sample":
            # do sample-wise compression
            sample = compression.compress(sample)

        # TODO: this can be replaced with hilbert curve or another locality-preserving flattening
        b = array_to_bytes(sample)

        chunk_gen = generate_chunks(b, chunk_size)
        chunk_names = []
        incomplete_chunk_names = []

        for chunk in chunk_gen:
            # TODO: chunk_name should be based on `global_chunk_index`
            chunk_name = "c%i" % local_chunk_index

            end_byte = len(chunk)  # end byte is based on the uncompressed chunk

            if len(chunk) >= chunk_size:
                if compression.subject == "chunk":
                    # TODO: add threshold for compressing (in case user specifies like 10gb chunk_size)
                    chunk = compression.compress(chunk)
            else:
                incomplete_chunk_names.append(chunk_name)

            chunk_names.append(chunk_name)
            chunk_key = os.path.join(key, chunk_name)
            storage[chunk_key] = chunk

            local_chunk_index += 1

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
