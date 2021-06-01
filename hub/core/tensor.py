import numpy as np

from hub.core.typing import StorageProvider

from hub.core.meta.tensor_meta import read_tensor_meta, write_tensor_meta, validate_tensor_meta
from hub.core.meta.index_map import read_index_map, write_index_map
from hub.util.keys import get_tensor_meta_key, get_index_map_key
from hub.util.array import normalize_and_batchify_shape
from hub.util.exceptions import TensorAlreadyExistsError, TensorMetaMismatchError, TensorNotFoundError

from hub.core.chunk_engine.read import sample_from_index_entry
from hub.core.chunk_engine.write import write_bytes

from .flatten import row_wise_to_bytes


def tensor_exists(key: str, storage: StorageProvider) -> bool:
    """A tensor exists if at the specified `key` and `storage` there is both a meta file and index map."""

    meta_key = get_tensor_meta_key(key)
    index_map_key = get_index_map_key(key)
    return meta_key in storage and index_map_key in storage


def create_tensor(key: str, storage: StorageProvider, meta: dict):
    """If a tensor does not exist, create a new one with the provided meta.

    Args:
        key (str): Key for where the chunks, index_map, and meta will be located in `storage` relative to it's root.
        storage (StorageProvider): StorageProvider that all tensor data is written to.
        meta (dict): Meta for the tensor. Required Properties:
            # TODO: fill in properties
            chunk_size (int): Desired length of chunks.
            dtype (str): Datatype for each sample.

    Raises: 
        TensorAlreadyExistsError: If a tensor defined with `key` already exists.
    """

    if tensor_exists(key, storage):
        raise TensorAlreadyExistsError(key)

    validate_tensor_meta(meta)

    write_tensor_meta(key, storage, meta)
    write_index_map(key, storage, [])


def add_samples_to_tensor(
    array: np.ndarray,
    key: str,
    storage: StorageProvider,
    batched: bool = False,
):
    """Create a new tensor (if one doesn't already exist), then chunk and write the given array to storage.

    For more on chunking, see the `generate_chunks` method.

    Args:
        array (np.ndarray): Array to be chunked/written. Batch axis (`array.shape[0]`) is optional, if `array` does have a
            batch axis, you should pass the argument `batched=True`.
        key (str): Key for where the chunks, index_map, and meta will be located in `storage` relative to it's root.
        storage (StorageProvider): StorageProvider for storing the chunks, index_map, and meta.

        batched (bool): If True, the provied `array`'s first axis (`shape[0]`) will be considered it's batch axis.
            If False, a new axis will be created with a size of 1 (`array.shape[0] == 1`). default=False
    """

    array = normalize_and_batchify_shape(array, batched=batched)

    if not tensor_exists(key, storage):
        raise Exception()  # TODO: exceptions.py

    index_map = read_index_map(key, storage)
    tensor_meta = read_tensor_meta(key, storage)
    _check_array_and_tensor_are_compatible(tensor_meta, array)

    # TODO: get the tobytes function from meta
    tobytes = row_wise_to_bytes

    array_length = array.shape[0]
    for i in range(array_length):
        sample = array[i]

        # TODO: we may want to call `tobytes` on `array` and call memoryview on that. this may depend on the access patterns we
        # choose to optimize for.
        b = memoryview(tobytes(sample))

        index_map_entry = write_bytes(
            b, key, tensor_meta["chunk_size"], storage, index_map
        )

        # shape per sample for dynamic tensors (TODO: if strictly fixed-size, store this in meta)
        index_map_entry["shape"] = sample.shape
        index_map.append(index_map_entry)

    tensor_meta["length"] += array_length

    write_tensor_meta(key, storage, tensor_meta)
    write_index_map(key, storage, index_map)


def read_samples_from_tensor(
    key: str,
    storage: StorageProvider,
    array_slice: slice = slice(None),
) -> np.ndarray:
    """Read (and unpack) samples from a tensor as an np.ndarray.

    Args:
        key (str): Key for where the chunks, index_map, and meta are located in `storage` relative to it's root.
        array_slice (slice): Slice that represents which samples to read. Default = slice representing all samples.
        storage (StorageProvider): StorageProvider for reading the chunks, index_map, and meta.

    Returns:
        np.ndarray: Array containing the sample(s) in the `array_slice` slice.
    """

    meta = read_tensor_meta(key, storage)
    index_map = read_index_map(key, storage)

    # TODO: read samples in parallel
    samples = []
    for index_entry in index_map[array_slice]:
        array = sample_from_index_entry(key, storage, index_entry, meta["dtype"])
        samples.append(array)

    return np.array(samples)


def _check_array_and_tensor_are_compatible(tensor_meta: dict, array: np.ndarray):
    """An array is considered incompatible with a tensor if the `tensor_meta` entries don't match the `array` properties.

    Args:
        tensor_meta (dict): Tensor meta containing the expected properties of `array`.
        array (np.ndarray): Candidate array to check compatibility with `tensor_meta`.

    Raises:
        TensorMetaMismatchError: When `array` properties do not match the `tensor_meta`'s exactly. Also when `len(array.shape)` != len(tensor_meta max/min shapes).
        NotImplementedError: When `array.shape` does not match for all samples. Dynamic shapes are not yet supported. (TODO)
    """

    if tensor_meta["dtype"] != array.dtype.name:
        raise TensorMetaMismatchError("dtype", tensor_meta["dtype"], array.dtype.name)

    sample_shape = array.shape[1:]
    if len(tensor_meta["min_shape"]) != len(sample_shape):
        raise TensorMetaMismatchError(
            "min_shape", tensor_meta["min_shape"], len(sample_shape)
        )
    if len(tensor_meta["max_shape"]) != len(sample_shape):
        raise TensorMetaMismatchError(
            "max_shape", tensor_meta["max_shape"], len(sample_shape)
        )

    # TODO: remove these once dynamic shapes are supported
    if not np.array_equal(tensor_meta["max_shape"], sample_shape):
        raise NotImplementedError("Dynamic shapes are not supported yet.")
    if not np.array_equal(tensor_meta["min_shape"], sample_shape):
        raise NotImplementedError("Dynamic shapes are not supported yet.")