import numpy as np
import pickle
from uuid import uuid1

from hub.core.chunk_engine import generate_chunks

from hub.core.typing import Provider
from typing import Any, Callable, List, Tuple

from .util import (
    array_to_bytes,
    normalize_and_batchify_shape,
    get_meta_key,
    get_index_map_key,
    get_chunk_key,
)


def write_array(
    array: np.ndarray,
    key: str,
    chunk_size: int,
    storage: Provider,
    batched: bool = False,
    tobytes: Callable[[np.ndarray], bytes] = array_to_bytes,
):
    """Chunk & write an array to storage.

    Args:
        array (np.ndarray): Array to be chunked/written. Batch axis (`array.shape[0]`) is optional, if `array` does have a
            batch dimension, you should pass `batched=True`.
        key (str): Key for where the chunks/index_map/meta will be located in `storage` relative to it's root.
        chunk_size (int): Each individual chunk will be assigned this many bytes maximum.
        storage (Provider): Provider for storing the chunks, index_map, & meta.
        batched (bool): If True, the provied `array`'s first axis (`shape[0]`) will be considered it's batch axis.
            If False, a new axis will be created with a size of 1 (`array.shape[0] == 1`). default=False
        tobytes (Callable): Must accept an `np.ndarray` as it's argument & return `bytes`.

    Raises:
        NotImplementedError: Do not use this function for writing to a key that already exists.
    """

    array = normalize_and_batchify_shape(array, batched=batched)

    meta_key = get_meta_key(key)
    index_map_key = get_index_map_key(key)
    if meta_key in storage or index_map_key in storage:
        raise NotImplementedError("Appending is not supported yet.")

    index_map = []
    meta = {
        "chunk_size": chunk_size,
        "dtype": array.dtype.name,
        "length": array.shape[0],
        "min_shape": array.shape[1:],
        "max_shape": array.shape[1:],
    }

    for i in range(array.shape[0]):
        sample = array[i]

        b = tobytes(sample)

        chunk_gen = generate_chunks(b, chunk_size)
        chunk_names = []
        incomplete_chunk_names = []

        for chunk in chunk_gen:
            chunk_name = _random_chunk_name()

            end_byte = len(chunk)  # end byte is based on the uncompressed chunk

            if len(chunk) < chunk_size:
                incomplete_chunk_names.append(chunk_name)

            chunk_names.append(chunk_name)
            chunk_key = get_chunk_key(key, chunk_name)
            storage[chunk_key] = chunk

        # TODO: encode index_map_entry as array instead of dictionary
        # TODO: encode shape into the sample's bytes instead of index_map
        index_map_entry = {
            "chunk_names": chunk_names,
            "incomplete_chunk_names": incomplete_chunk_names,
            "start_byte": 0,
            "end_byte": end_byte,
            "shape": sample.shape,  # shape per sample for dynamic tensors (if strictly fixed-size, store this in meta)
        }
        index_map.append(index_map_entry)

    # TODO: don't use pickle
    storage[meta_key] = pickle.dumps(meta)
    storage[index_map_key] = pickle.dumps(index_map)


def _random_chunk_name() -> str:
    return str(uuid1())
