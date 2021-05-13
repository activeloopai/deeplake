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
    """Chunk and write an array to storage.

    For more on chunking, see the `generate_chunks` method.

    Args:
        array (np.ndarray): Array to be chunked/written. Batch axis (`array.shape[0]`) is optional, if `array` does have a
            batch dimension, you should pass the argument `batched=True`.
        key (str): Key for where the chunks, index_map, and meta will be located in `storage` relative to it's root.
        chunk_size (int): Desired length of each chunk.
        storage (Provider): Provider for storing the chunks, index_map, and meta.
        batched (bool): If True, the provied `array`'s first axis (`shape[0]`) will be considered it's batch axis.
            If False, a new axis will be created with a size of 1 (`array.shape[0] == 1`). default=False
        tobytes (Callable): Must accept an `np.ndarray` as it's argument and return `bytes`.

    Raises:
        NotImplementedError: Do not use this function for writing to a key that already exists.
    """

    array = normalize_and_batchify_shape(array, batched=batched)

    meta_key = get_meta_key(key)
    index_map_key = get_index_map_key(key)
    if meta_key in storage or index_map_key in storage:
        # TODO: when appending is done change this error
        raise NotImplementedError("Appending is not supported yet.")

    index_map: List[dict] = []
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

        index_map_entry = write_bytes(b, key, chunk_size, storage, index_map)

        # shape per sample for dynamic tensors (TODO: if strictly fixed-size, store this in meta)
        index_map_entry["shape"] = sample.shape
        index_map.append(index_map_entry)

    # TODO: don't use pickle
    storage[meta_key] = pickle.dumps(meta)
    storage[index_map_key] = pickle.dumps(index_map)


def write_bytes(
    b: bytes, key: str, chunk_size: int, storage: Provider, index_map: List[dict]
) -> dict:
    """For internal use only. Chunk and write bytes to storage and return the index_map entry.
    The provided bytes are treated as a single sample.

    Args:
        b (bytes): Bytes to be chunked/written. `b` is considered to be 1 sample and will be chunked according
            to `chunk_size`.
        key (str): Key for where the index_map, and meta are located in `storage` relative to it's root. A subdirectory
            is created under this `key` (defined in `constants.py`), which is where the chunks will be stored.
        chunk_size (int): Desired length of each chunk.
        storage (Provider): Provider for storing the chunks, index_map, and meta.
        index_map (list): List of dictionaries that represent each sample. An entry for `index_map` is returned
            but not appended to `index_map`.

    Returns:
        dict: Index map entry (note: it does not get appended to the `index_map` argument). Dictionary keys:
            chunk_names: Sequential list of names of chunks that were created.
            start_byte: Start byte for this sample. Will be 0 if no previous chunks exist, otherwise will
                be set to the length of the last chunk before writing.
            end_byte: End byte for this sample. Will be equal to the length of the last chunk written to.
    """

    last_chunk_name, last_chunk = _get_last_chunk(key, index_map, storage)

    bllc = 0
    extend_last_chunk = False
    if len(index_map) > 0 and len(last_chunk) < chunk_size:
        bllc = chunk_size - len(last_chunk)
        extend_last_chunk = True

    chunk_generator = generate_chunks(b, chunk_size, bytes_left_in_last_chunk=bllc)

    chunk_names = []
    start_byte = 0
    for chunk in chunk_generator:
        if extend_last_chunk:
            chunk_name = last_chunk_name
            last_chunk_bytearray = bytearray(last_chunk)
            last_chunk_bytearray.extend(chunk)
            chunk = bytes(last_chunk_bytearray)
            start_byte = index_map[-1]["end_byte"]

            if len(chunk) >= chunk_size:
                extend_last_chunk = False
        else:
            chunk_name = _random_chunk_name()

        end_byte = len(chunk)

        chunk_key = get_chunk_key(key, chunk_name)
        storage[chunk_key] = chunk

        chunk_names.append(chunk_name)

        last_chunk = chunk
        last_chunk_name = chunk_name

    # TODO: encode index_map_entry as array instead of dictionary
    # TODO: encode shape into the sample's bytes instead of index_map
    index_map_entry = {
        "chunk_names": chunk_names,
        "start_byte": start_byte,
        "end_byte": end_byte,
    }

    return index_map_entry


def _get_last_chunk(
    key: str, index_map: List[dict], storage: Provider
) -> Tuple[str, bytes]:
    """For internal use only. Retrieves the name and bytes for the last chunk.

    Args:
        key (str): Key for where the chunks are located in `storage` relative to it's root.
        index_map (list): List of dictionaries that maps each sample to the `chunk_names`, `start_byte`, and `end_byte`.
        storage (Provider): Provider where the chunks are stored.

    Returns:
        str: Name of the last chunk. If the last chunk doesn't exist, returns an empty string.
        bytes: Content of the last chunk. If the last chunk doesn't exist, returns empty bytes.
    """

    last_chunk_name = ""
    last_chunk = bytes()
    if len(index_map) > 0:
        last_index_map_entry = index_map[-1]
        last_chunk_name = last_index_map_entry["chunk_names"][-1]
        last_chunk_key = get_chunk_key(key, last_chunk_name)
        last_chunk = storage[last_chunk_key]
    return last_chunk_name, last_chunk


def _random_chunk_name() -> str:
    return str(uuid1())
