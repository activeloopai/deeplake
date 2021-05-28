import numpy as np
import pickle  # TODO: NEVER USE PICKLE
from uuid import uuid1

from hub.core.chunk_engine import generate_chunks
from hub.constants import META_FILENAME, DEFAULT_CHUNK_SIZE

from hub.core.typing import StorageProvider
from typing import Any, Callable, List, Tuple

from .read import read_index_map, read_tensor_meta, tensor_exists
from .flatten import row_wise_to_bytes


from hub.util.keys import get_meta_key, get_index_map_key, get_chunk_key
from hub.util.array import normalize_and_batchify_shape
from hub.util.exceptions import (
    MetaMismatchError,
)


def write_tensor_meta(key: str, storage: StorageProvider, meta: dict):
    storage[get_meta_key(key)] = pickle.dumps(meta)


def write_index_map(key: str, storage: StorageProvider, index_map: list):
    index_map_key = get_index_map_key(key)
    storage[index_map_key] = pickle.dumps(index_map)


def write_dataset_meta(storage: StorageProvider, meta: dict):
    storage[META_FILENAME] = pickle.dumps(meta)


def add_samples_to_tensor(
    array: np.ndarray,
    key: str,
    storage: StorageProvider,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    batched: bool = False,
):

    """Create a new tensor (if one doesn't already exist), then chunk and write the given array to storage.
    For writing an array to an already existing tensor, use `append_array`.

    For more on chunking, see the `generate_chunks` method.

    Args:
        array (np.ndarray): Array to be chunked/written. Batch axis (`array.shape[0]`) is optional, if `array` does have a
            batch axis, you should pass the argument `batched=True`.
        key (str): Key for where the chunks, index_map, and meta will be located in `storage` relative to it's root.
        storage (StorageProvider): StorageProvider for storing the chunks, index_map, and meta.
        chunk_size (int): Desired length of each chunk.
        batched (bool): If True, the provied `array`'s first axis (`shape[0]`) will be considered it's batch axis.
            If False, a new axis will be created with a size of 1 (`array.shape[0] == 1`). default=False
    """

    array = normalize_and_batchify_shape(array, batched=batched)

    if tensor_exists(key, storage):
        index_map = read_index_map(key, storage)
        tensor_meta = read_tensor_meta(key, storage)
        _check_array_and_tensor_are_compatible(tensor_meta, array, chunk_size)

    else:
        index_map: List[dict] = []
        tensor_meta = {
            "chunk_size": chunk_size,
            "dtype": array.dtype.name,
            "length": array.shape[0],
            "min_shape": tuple(array.shape[1:]),
            "max_shape": tuple(array.shape[1:]),
            # TODO: add entry in meta for which tobytes function is used and handle mismatch versions for this
        }

    # TODO: get the tobytes function from meta
    tobytes = row_wise_to_bytes

    for i in range(array.shape[0]):
        sample = array[i]
        b = memoryview(tobytes(sample))

        index_map_entry = write_bytes(b, key, chunk_size, storage, index_map)

        # shape per sample for dynamic tensors (TODO: if strictly fixed-size, store this in meta)
        index_map_entry["shape"] = sample.shape
        index_map.append(index_map_entry)

    write_tensor_meta(key, storage, tensor_meta)
    write_index_map(key, storage, index_map)


def write_bytes(
    b: memoryview,
    key: str,
    chunk_size: int,
    storage: StorageProvider,
    index_map: List[dict],
) -> dict:
    """For internal use only. Chunk and write bytes to storage and return the index_map entry.
    The provided bytes are treated as a single sample.

    Args:
        b (memoryview): Bytes (as memoryview) to be chunked/written. `b` is considered to be 1 sample and will be chunked according
            to `chunk_size`.
        key (str): Key for where the index_map, and meta are located in `storage` relative to it's root. A subdirectory
            is created under this `key` (defined in `constants.py`), which is where the chunks will be stored.
        chunk_size (int): Desired length of each chunk.
        storage (StorageProvider): StorageProvider for storing the chunks, index_map, and meta.
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
        # use bytearray for concatenation (fastest method)
        last_chunk = bytearray(last_chunk)  # type: ignore
        extend_last_chunk = True

    chunk_generator = generate_chunks(b, chunk_size, bytes_left_in_last_chunk=bllc)

    chunk_names = []
    start_byte = 0
    for chunk in chunk_generator:
        if extend_last_chunk:
            chunk_name = last_chunk_name

            last_chunk += chunk  # type: ignore
            chunk = memoryview(last_chunk)

            start_byte = index_map[-1]["end_byte"]

            if len(chunk) >= chunk_size:
                extend_last_chunk = False
        else:
            chunk_name = _random_chunk_name()

        end_byte = len(chunk)

        chunk_key = get_chunk_key(key, chunk_name)
        storage[chunk_key] = chunk

        chunk_names.append(chunk_name)

        last_chunk = memoryview(chunk)
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
    key: str, index_map: List[dict], storage: StorageProvider
) -> Tuple[str, memoryview]:
    """For internal use only. Retrieves the name and memoryview of bytes for the last chunk.

    Args:
        key (str): Key for where the chunks are located in `storage` relative to it's root.
        index_map (list): List of dictionaries that maps each sample to the `chunk_names`, `start_byte`, and `end_byte`.
        storage (StorageProvider): StorageProvider where the chunks are stored.

    Returns:
        str: Name of the last chunk. If the last chunk doesn't exist, returns an empty string.
        memoryview: Content of the last chunk. If the last chunk doesn't exist, returns empty memoryview of bytes.
    """

    if len(index_map) > 0:
        last_index_map_entry = index_map[-1]
        last_chunk_name = last_index_map_entry["chunk_names"][-1]
        last_chunk_key = get_chunk_key(key, last_chunk_name)
        last_chunk = memoryview(storage[last_chunk_key])
        return last_chunk_name, last_chunk
    return "", memoryview(bytes())


def _random_chunk_name() -> str:
    return str(uuid1())


def _check_array_and_tensor_are_compatible(
    meta: dict, array: np.ndarray, chunk_size: int
):
    if meta["dtype"] != array.dtype.name:
        raise MetaMismatchError("dtype", meta["dtype"], array.dtype.name)

    sample_shape = array.shape[1:]
    if len(meta["min_shape"]) != len(sample_shape):
        raise MetaMismatchError("min_shape", meta["min_shape"], len(sample_shape))
    if len(meta["max_shape"]) != len(sample_shape):
        raise MetaMismatchError("max_shape", meta["max_shape"], len(sample_shape))

    if chunk_size is not None and chunk_size != meta["chunk_size"]:
        raise MetaMismatchError("chunk_size", meta["chunk_size"], chunk_size)

    # TODO: remove these once dynamic shapes are supported
    if not np.array_equal(meta["max_shape"], sample_shape):
        raise NotImplementedError("Dynamic shapes are not supported yet.")
    if not np.array_equal(meta["min_shape"], sample_shape):
        raise NotImplementedError("Dynamic shapes are not supported yet.")
