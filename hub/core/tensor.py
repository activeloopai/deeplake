from hub.htypes import DEFAULT_HTYPE
from hub.core.meta.tensor_meta import TensorMeta
from hub.core.meta.index_meta import IndexMeta
from hub.core.index import Index
from typing import List, Tuple, Union
import numpy as np

from hub.core.chunk_engine.read import sample_from_index_entry
from hub.core.chunk_engine.write import write_array
from hub.util.keys import get_index_meta_key, get_tensor_meta_key
from hub.core.typing import StorageProvider
from hub.util.exceptions import (
    DynamicTensorNumpyError,
    TensorAlreadyExistsError,
    TensorDoesNotExistError,
)


def tensor_exists(key: str, storage: StorageProvider) -> bool:
    """A tensor exists if at the specified `key` and `storage` there is both a tensor meta file and index map."""

    meta_key = get_tensor_meta_key(key)
    index_meta_key = get_index_meta_key(key)
    return meta_key in storage and index_meta_key in storage


def create_tensor(
    key: str,
    storage: StorageProvider,
    htype: str = DEFAULT_HTYPE,
    **kwargs,
):
    """If a tensor does not exist, create a new one with the provided meta.

    Args:
        key (str): Key for where the chunks, index_meta, and tensor_meta will be located in `storage` relative to it's root.
        storage (StorageProvider): StorageProvider that all tensor data is written to.
        htype (str): Htype is how the default tensor metadata is defined.
        **kwargs: `htype` defaults can be overridden by passing any of the compatible parameters.
            To see all `htype`s and their correspondent arguments, check out `hub/htypes.py`.

    Raises:
        TensorAlreadyExistsError: If a tensor defined with `key` already exists.
    """

    if tensor_exists(key, storage):
        raise TensorAlreadyExistsError(key)

    TensorMeta.create(key, storage, htype=htype, **kwargs)
    IndexMeta.create(key, storage)


def _get_metas_from_kwargs(
    key: str, storage: StorageProvider, **kwargs
) -> Tuple[TensorMeta, IndexMeta]:
    if "tensor_meta" in kwargs:
        tensor_meta = kwargs["tensor_meta"]
    else:
        tensor_meta = TensorMeta.load(key, storage)

    if "index_meta" in kwargs:
        index_meta = kwargs["index_meta"]
    else:
        index_meta = IndexMeta.load(key, storage)

    return tensor_meta, index_meta


def append_tensor(array: np.ndarray, key: str, storage: StorageProvider, **kwargs):
    """Append to an existing tensor with an array. This array will be chunked and sent to `storage`.

    For more on chunking, see the `generate_chunks` method.

    Args:
        array (np.ndarray): Array to be chunked/written. This array will be considered as 1 sample.
        key (str): Key for where the chunks, index_meta, and meta will be located in `storage` relative to it's root.
        storage (StorageProvider): StorageProvider for storing the chunks, index_meta, and meta.
        **kwargs:
            tensor_meta (TensorMeta): Optionally provide a `TensorMeta`. If not provided, it will be loaded from `storage`.
            index_meta (IndexMeta): Optionally provide an `IndexMeta`. If not provided, it will be loaded from `storage`.

    Raises:
        TensorDoesNotExistError: If a tensor at `key` does not exist. A tensor must be created first using
            `create_tensor(...)`.
    """

    # append is guaranteed to NOT have a batch axis
    array = np.expand_dims(array, axis=0)
    extend_tensor(array, key, storage, **kwargs)


def extend_tensor(array: np.ndarray, key: str, storage: StorageProvider, **kwargs):
    """Extend an existing tensor with an array. This array will be chunked and sent to `storage`.

    For more on chunking, see the `generate_chunks` method.

    Args:
        array (np.ndarray): Array to be chunked/written. This array will be considered as 1 sample.
        key (str): Key for where the chunks, index_meta, and meta will be located in `storage` relative to it's root.
        storage (StorageProvider): StorageProvider for storing the chunks, index_meta, and meta.
        **kwargs:
            tensor_meta (TensorMeta): Optionally provide a `TensorMeta`. If not provided, it will be loaded from `storage`.
            index_meta (IndexMeta): Optionally provide an `IndexMeta`. If not provided, it will be loaded from `storage`.

    Raises:
        ValueError: If `array` has <= 1 axes.
        TensorDoesNotExistError: If a tensor at `key` does not exist. A tensor must be created first using
            `create_tensor(...)`.
    """

    if len(array.shape) < 1:
        raise ValueError(
            f"An array with shape={array.shape} cannot be used to extend because it's shape length is < 1."
        )

    if not tensor_exists(key, storage):
        raise TensorDoesNotExistError(key)

    tensor_meta, index_meta = _get_metas_from_kwargs(key, storage, **kwargs)

    # extend is guaranteed to have a batch axis
    tensor_meta.check_batch_is_compatible(array)

    write_array(array, key, storage, tensor_meta, index_meta)


def read_samples_from_tensor(
    key: str,
    storage: StorageProvider,
    index: Index = Index(),
    aslist: bool = False,
) -> Union[np.ndarray, List[np.ndarray]]:
    """Read (and unpack) samples from a tensor as an np.ndarray.

    Args:
        key (str): Key for where the chunks, index_meta, and meta are located in `storage` relative to it's root.
        storage (StorageProvider): StorageProvider for reading the chunks, index_meta, and meta.
        index (Index): Index that represents which samples to read.
        aslist (bool): If True, a list of np.ndarrays will be returned. Helpful for dynamic tensors.
            If False, a single np.ndarray will be returned unless the samples are dynamically shaped, in which case
            an error is raised.

    Raises:
        DynamicTensorNumpyError: If reading a dynamically-shaped array slice without `aslist=True`.

    Returns:
        np.ndarray: Array containing the sample(s) in the `array_slice` slice.
    """

    index_meta = IndexMeta.load(key, storage)
    tensor_meta = TensorMeta.load(key, storage)

    index_entries = [
        index_meta.entries[i] for i in index.values[0].indices(len(index_meta.entries))
    ]

    # TODO: read samples in parallel
    samples = []
    for i, index_entry in enumerate(index_entries):
        shape = index_entry["shape"]

        # check if all samples are the same shape
        last_shape = index_entries[i - 1]["shape"]
        if not aslist and shape != last_shape:
            raise DynamicTensorNumpyError(key, index)

        array = sample_from_index_entry(key, storage, index_entry, tensor_meta.dtype)
        samples.append(array)

    if aslist:
        if index.values[0].subscriptable():
            return samples
        else:
            return samples[0]

    return index.apply(np.array(samples))
