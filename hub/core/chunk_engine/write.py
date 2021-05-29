import numpy as np
import json
from uuid import uuid1

from hub.core.chunk_engine import generate_chunks
from hub.constants import META_FILENAME, DEFAULT_CHUNK_SIZE

from hub.core.typing import StorageProvider
from typing import Any, Callable, List, Tuple

from .read import read_index_map, read_tensor_meta, key_exists
from .flatten import row_wise_to_bytes


from hub.util.keys import get_meta_key, get_index_map_key, get_chunk_key
from hub.util.array import normalize_and_batchify_shape
from hub.util.exceptions import (
    KeyAlreadyExistsError,
    KeyDoesNotExistError,
    MetaMismatchError,
)


def _listify(shape: Tuple):
    shapeArray = np.asarray(shape)
    return shapeArray.tolist()


def write_tensor_meta(key: str, storage: StorageProvider, meta: dict):
    meta["min_shape"] = _listify(meta["min_shape"])
    meta["max_shape"] = _listify(meta["max_shape"])
    storage[get_meta_key(key)] = bytes(json.dumps(meta), "utf-8")


def write_index_map(key: str, storage: StorageProvider, index_map: list):
    index_map_key = get_index_map_key(key)
    storage[index_map_key] = bytes(json.dumps(index_map), "utf-8")


def write_dataset_meta(storage: StorageProvider, meta: dict):
    storage[META_FILENAME] = bytes(json.dumps(meta), "utf-8")


def write_array(
    array: np.ndarray,
    key: str,
    storage: StorageProvider,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    batched: bool = False,
):
    """Create a new tensor, then chunk and write the given array to storage. For writing an array to an already existing tensor, use `append_array`.

    For more on chunking, see the `generate_chunks` method.

    Args:
        array (np.ndarray): Array to be chunked/written. Batch axis (`array.shape[0]`) is optional, if `array` does have a
            batch axis, you should pass the argument `batched=True`.
        key (str): Key for where the chunks, index_map, and meta will be located in `storage` relative to it's root.
        storage (StorageProvider): StorageProvider for storing the chunks, index_map, and meta.
        chunk_size (int): Desired length of each chunk.
        batched (bool): If True, the provied `array`'s first axis (`shape[0]`) will be considered it's batch axis.
            If False, a new axis will be created with a size of 1 (`array.shape[0] == 1`). default=False

    Raises:
        KeyAlreadyExistsError: If trying to write to a tensor that already exists in `storage` under `key`.
    """

    if key_exists(key, storage):
        raise KeyAlreadyExistsError(key, "Use `append_array`.")

    array = normalize_and_batchify_shape(array, batched=batched)

    index_map: List[dict] = []
    meta = {
        "chunk_size": chunk_size,
        "dtype": array.dtype.name,
        "length": array.shape[0],
        "min_shape": tuple(array.shape[1:]),
        "max_shape": tuple(array.shape[1:]),
        # TODO: add entry in meta for which tobytes function is used and handle mismatch versions for this
    }

    write_samples(array, key, storage, meta, index_map)

    write_tensor_meta(key, storage, meta)
    write_index_map(key, storage, index_map)


def append_array(
    array: np.ndarray,
    key: str,
    storage: StorageProvider,
    batched: bool = False,
):
    """Chunk and write the given array to an already existing tensor in storage. For writing an array to a tensor that does not exist, use `write_array`.

    For more on chunking, see the `generate_chunks` method.

    Args:
        array (np.ndarray): Array to be chunked/written. Batch axis (`array.shape[0]`) is optional, if `array` does have a
            batch axis, you should pass the argument `batched=True`.
        key (str): Key for where the chunks, index_map, and meta are located in `storage` relative to it's root.
        storage (StorageProvider): StorageProvider where the chunks, index_map, and meta are stored.
        batched (bool): If True, the provied `array`'s first axis (`shape[0]`) will be considered it's batch axis.
            If False, a new axis will be created with a size of 1 (`array.shape[0] == 1`). default=False

    Raises:
        KeyDoesNotExistError: If trying to append to a tensor that does not exist in `storage` under `key`.
    """

    if not key_exists(key, storage):
        raise KeyDoesNotExistError(key, "Use `write_array`.")

    array = normalize_and_batchify_shape(array, batched=batched)

    index_map = read_index_map(key, storage)
    meta = read_tensor_meta(key, storage)

    _check_if_meta_is_compatible_with_array(meta, array)

    write_samples(array, key, storage, meta, index_map)

    # TODO: write tests to check if min/max shape is properly set (after we add dynamic shapes)
    # TODO: move this into function (especially tuple wrapping)
    sample_shape = array.shape[1:]
    min_shape = np.minimum(meta["min_shape"], sample_shape)
    max_shape = np.maximum(meta["max_shape"], sample_shape)
    meta["min_shape"] = tuple(min_shape)
    meta["max_shape"] = tuple(max_shape)

    write_tensor_meta(key, storage, meta)
    write_index_map(key, storage, index_map)


def write_samples(
    array: np.ndarray,
    key: str,
    storage: StorageProvider,
    meta: dict,
    index_map: List[dict],
):
    chunk_size = meta["chunk_size"]

    # TODO: get the tobytes function from meta
    tobytes = row_wise_to_bytes

    for i in range(array.shape[0]):
        sample = array[i]
        b = tobytes(sample)

        index_map_entry = write_bytes(b, key, chunk_size, storage, index_map)

        # shape per sample for dynamic tensors (TODO: if strictly fixed-size, store this in meta)
        index_map_entry["shape"] = sample.shape
        index_map.append(index_map_entry)


def write_bytes(
    b: bytes, key: str, chunk_size: int, storage: StorageProvider, index_map: List[dict]
) -> dict:
    """For internal use only. Chunk and write bytes to storage and return the index_map entry.
    The provided bytes are treated as a single sample.

    Args:
        b (bytes): Bytes to be chunked/written. `b` is considered to be 1 sample and will be chunked according
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
    key: str, index_map: List[dict], storage: StorageProvider
) -> Tuple[str, bytes]:
    """For internal use only. Retrieves the name and bytes for the last chunk.

    Args:
        key (str): Key for where the chunks are located in `storage` relative to it's root.
        index_map (list): List of dictionaries that maps each sample to the `chunk_names`, `start_byte`, and `end_byte`.
        storage (StorageProvider): StorageProvider where the chunks are stored.

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


def _check_if_meta_is_compatible_with_array(meta: dict, array: np.ndarray):
    if meta["dtype"] != array.dtype.name:
        raise MetaMismatchError("dtype", meta["dtype"], array.dtype.name)

    sample_shape = array.shape[1:]
    if len(meta["min_shape"]) != len(sample_shape):
        raise MetaMismatchError("min_shape", meta["min_shape"], len(sample_shape))
    if len(meta["max_shape"]) != len(sample_shape):
        raise MetaMismatchError("max_shape", meta["max_shape"], len(sample_shape))

    # TODO: remove these once dynamic shapes are supported
    if not np.array_equal(meta["max_shape"], sample_shape):
        raise NotImplementedError("Dynamic shapes are not supported yet.")
    if not np.array_equal(meta["min_shape"], sample_shape):
        raise NotImplementedError("Dynamic shapes are not supported yet.")
