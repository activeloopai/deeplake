import os
import numpy as np
import pickle

from typing import Any, Callable, List, Tuple

from hub.core.chunk_engine import generate_chunks
from .util import array_to_bytes, index_map_entry_to_bytes, normalize_and_batchify_shape
from .dummy_util import MemoryProvider


def chunk_and_write_array(
    array: np.ndarray,
    key: str,
    compression,
    chunk_size: int,
    storage: MemoryProvider,
    batched: bool = False,
):
    """
    Chunk, & write array to `storage`.
    """

    # TODO: for most efficiency, we should try to use `batched` as often as possible.

    # TODO: validate array shape (no 0s in shape)
    array = normalize_and_batchify_shape(array, batched=batched)

    # TODO: validate meta matches tensor meta where it needs to (like dtype or strict-shape)
    # TODO: update existing meta. for example, if meta["length"] already exists, we will need to add instead of set
    meta = {
        "dtype": array.dtype,
        "length": array.shape[0],
        "compression": compression.__name__,
    }

    index_map = []
    for i in range(array.shape[0]):
        sample = array[i]
        if compression.subject == "sample":
            # do sample-wise compression
            sample = compression.compress(sample)

        # TODO: this can be replaced with hilbert curve or something
        b = array_to_bytes(sample)
        start_chunk, end_chunk = chunk_and_write_bytes(
            b,
            key=key,
            compression=compression,
            chunk_size=chunk_size,
            storage=storage,
        )

        # TODO: keep track of `sample.shape` over time & add the max_shape:min_shape interval into meta.json for easy queries

        # TODO: encode index_map_entry as array instead of dictionary
        index_map.append(
            {
                "start_chunk": start_chunk,
                "end_chunk": end_chunk,
                "shape": sample.shape,  # shape per sample for dynamic tensors (if strictly fixed-size, store this in meta)
            }
        )

    # TODO: don't use pickle for index_map/meta
    # TODO: chunk index_map
    index_map_key = os.path.join(key, "index_map")
    storage[index_map_key] = pickle.dumps(index_map)

    meta_key = os.path.join(key, "meta.json")
    storage[meta_key] = pickle.dumps(meta)


def chunk_and_write_bytes(
    b: bytes,
    key: str,
    compression,
    chunk_size: int,
    storage: MemoryProvider,
    use_index_map: bool = True,
) -> Tuple[int, int]:
    """
    Chunk, & write bytes to `storage`.
    """

    bllc = 0  # TODO
    chunk_gen = generate_chunks(b, chunk_size, bytes_left_in_last_chunk=bllc)

    for local_chunk_index, chunk in enumerate(chunk_gen):
        # TODO: get global_chunk_index (don't just use local_chunk_index)

        full_chunk = len(chunk) == chunk_size

        chunk_name = "c%i" % local_chunk_index

        # TODO: fill previous chunk if it is incomplete
        # TODO: after previous chunk is fully filled, compress

        # TODO: add threshold for compressing (in case user specifies like 10gb chunk_size)
        if full_chunk and compression.subject == "chunk":
            # only compress if it is a full chunk

            chunk = compression.compress(chunk)
        else:
            chunk_name += "_incomplete"

        chunk_key = os.path.join(key, chunk_name)
        storage[chunk_key] = chunk

    # TODO global start/end chunk instead of local
    start_chunk = 0
    end_chunk = local_chunk_index
    return start_chunk, end_chunk
