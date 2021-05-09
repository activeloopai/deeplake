import os
import numpy as np
import pickle

from typing import Any, Callable, List, Tuple

from hub.core.chunk_engine import generate_chunks

from .meta import has_meta, get_meta, set_meta, default_meta
from .index_map import has_index_map, get_index_map, set_index_map, default_index_map
from .util import array_to_bytes, index_map_entry_to_bytes, normalize_and_batchify_shape


def get_and_validate_meta(key, storage, array, compression):
    if has_meta(key, storage):
        meta = get_meta(key, storage)

        # TODO: validate meta matches tensor meta where it needs to (like dtype or strict-shape)

        # TODO: exceptions.py
        if meta["dtype"] != array.dtype.name:
            raise Exception()
        if meta["compression"] != compression.__name__:
            raise Exception()
        print(meta)
        meta["length"] += array.shape[0]
        print(meta)
    else:
        meta = default_meta()
        meta.update(
            {
                "dtype": array.dtype.name,
                "length": array.shape[0],
                "compression": compression.__name__,
            }
        )

    print(meta, array.shape)

    return meta


def chunk_and_write_array(
    array: np.ndarray,
    key: str,
    compression,
    chunk_size: int,
    storage,
    batched: bool = False,
):
    """
    Chunk, & write array to `storage`.
    """

    # TODO: for most efficiency, we should try to use `batched` as often as possible.

    # TODO: validate array shape (no 0s in shape)
    array = normalize_and_batchify_shape(array, batched=batched)

    meta = get_and_validate_meta(key, storage, array, compression)

    # TODO: update existing meta. for example, if meta["length"] already exists, we will need to add instead of set

    local_chunk_index = 0

    # TODO: move into function:
    if has_index_map(key, storage):
        index_map = get_index_map(key, storage)
    else:
        index_map = default_index_map()

    for i in range(array.shape[0]):
        sample = array[i]
        if compression.subject == "sample":
            # do sample-wise compression
            sample = compression.compress(sample)

        # TODO: this can be replaced with hilbert curve or something
        b = array_to_bytes(sample)

        """chunk & write bytes"""
        bllc = 0  # TODO
        start_byte = 0  # TODO
        chunk_gen = generate_chunks(b, chunk_size, bytes_left_in_last_chunk=bllc)
        chunk_names = []
        incomplete_chunk_names = []
        end_byte = None

        for chunk in chunk_gen:
            # TODO: chunk_name should be based on `global_chunk_index`
            chunk_name = "c%i" % local_chunk_index

            # TODO: fill previous chunk if it is incomplete (compress if filled)

            if compression.subject == "chunk":
                # TODO: add threshold for compressing (in case user specifies like 10gb chunk_size)

                if len(chunk) >= chunk_size:
                    # only compress if it is a full chunk
                    end_byte = len(chunk)  # end byte is based on the uncompressed chunk
                    chunk = compression.compress(chunk)
                else:
                    incomplete_chunk_names.append(chunk_name)
                    end_byte = len(chunk)

            chunk_names.append(chunk_name)
            chunk_key = os.path.join(key, chunk_name)
            storage[chunk_key] = chunk

            local_chunk_index += 1

        """"""

        # TODO: make note of incomplete chunks
        # for chunk_name in incomplete_chunk_names:
        # storage[os

        # TODO: keep track of `sample.shape` over time & add the max_shape:min_shape interval into meta.json for easy queries

        # TODO: encode index_map_entry as array instead of dictionary
        index_map.append(
            {
                "chunk_names": chunk_names,
                "incomplete_chunk_names": incomplete_chunk_names,
                "start_byte": start_byte,
                "end_byte": end_byte,
                "shape": sample.shape,  # shape per sample for dynamic tensors (if strictly fixed-size, store this in meta)
            }
        )

    # TODO: chunk index_map
    set_index_map(key, storage, index_map)

    # update meta after everything is done
    set_meta(key, storage, meta)
