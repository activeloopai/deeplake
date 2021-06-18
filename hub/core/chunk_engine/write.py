from hub.constants import CHUNK_MAX_SIZE, CHUNK_MIN_TARGET
from hub.core.meta.tensor_meta import TensorMeta
import numpy as np
from hub.core.meta.index_meta import IndexMeta
from typing import List, Optional, Tuple, Union
from uuid import uuid1

from hub.core.typing import StorageProvider
from hub.util.keys import get_chunk_key
from math import ceil

from .flatten import row_wise_to_bytes


def write_array(
    array: np.ndarray,
    key: str,
    storage: StorageProvider,
    tensor_meta: TensorMeta,
    index_meta: IndexMeta,
):
    """Chunk and write an array to storage, also updates `index_meta`/`tensor_meta`. The provided array is treated as a batch of samples.

    Args:
        array (np.ndarray): Batched array to be chunked/written.
        key (str): Key for where the index_meta and tensor_meta are located in `storage` relative to it's root.
            A subdirectory is created under this `key` (defined in `constants.py`), which is where the chunks will be
            stored.
        storage (StorageProvider): StorageProvider for storing the chunks, index_meta, and tensor_meta.
        tensor_meta (TensorMeta): TensorMeta object that will be written to.
        index_meta (IndexMeta): IndexMeta object that will be written to.
    """

    # TODO: get the tobytes function from meta
    tobytes = row_wise_to_bytes

    num_samples = len(array)

    for i in range(num_samples):
        sample = array[i]
        extra_sample_meta = {"shape": sample.shape}

        if 0 in sample.shape:
            write_empty_sample(index_meta, extra_sample_meta=extra_sample_meta)

        else:
            # TODO: we may want to call `tobytes` on `array` and call memoryview on that. this may depend on the access patterns we
            # choose to optimize for.
            b = memoryview(tobytes(sample))
            write_bytes(
                b,
                key,
                storage,
                tensor_meta,
                index_meta,
                extra_sample_meta=extra_sample_meta,  # TODO: use kwargs
            )

        tensor_meta.update_with_sample(sample)

    tensor_meta.length += num_samples


def write_empty_sample(index_meta, extra_sample_meta: dict = {}):
    """Simply adds an entry to `index_map` that symbolizes an empty array."""

    index_meta.add_entry(chunk_names=[], start_byte=0, end_byte=0, **extra_sample_meta)


def write_bytes(
    content: memoryview,
    key: str,
    storage: StorageProvider,
    tensor_meta: TensorMeta,
    index_meta: IndexMeta,
    extra_sample_meta: dict = {},
):
    """Chunk and write bytes to storage, also updates `index_meta`/`tensor_meta`. The provided bytes are treated as a single sample.

    Args:
        content (memoryview): Bytes (as memoryview) to be chunked/written. Considered to be a single sample.
        key (str): Key for where the index_meta, and tensor_meta are located in `storage` relative to its root.
            A subdirectory is created under this `key` (defined in `constants.py`), which is where the chunks will be
            stored.
        storage (StorageProvider): StorageProvider for storing the chunks, index_meta, and tensor_meta.
        tensor_meta (TensorMeta): TensorMeta object that will be written to.
        index_meta (IndexMeta): IndexMeta object that will be written to.
        extra_sample_meta (dict): By default `chunk_names`, `start_byte`, and `end_byte` are written, however
            `IndexMeta.add_entry` supports more parameters than this. Anything passed in this dict will also be used
            to call `IndexMeta.add_entry`.
    """
    # TODO pass CHUNK_MIN, CHUNK_MAX and read from tensor_meta instead of using constants
    last_chunk_name, last_chunk = _get_last_chunk(key, storage, index_meta)
    start_byte = 0
    chunk_names: List[str] = []

    if _chunk_has_space(last_chunk):
        last_chunk_size = len(last_chunk)
        chunk_ct_content = _min_chunk_ct_for_data_size(len(content))

        extra_bytes = min(len(content), CHUNK_MAX_SIZE - last_chunk_size)
        combined_chunk_ct = _min_chunk_ct_for_data_size(len(content) + last_chunk_size)

        if combined_chunk_ct == chunk_ct_content:  # combine if count is same
            start_byte = index_meta.entries[-1]["end_byte"]
            end_byte = start_byte + extra_bytes

            chunk_content = bytearray(last_chunk) + content[0:extra_bytes]
            _write_chunk(chunk_content, storage, chunk_names, key, last_chunk_name)

            content = content[extra_bytes:]

    while len(content) > 0:
        end_byte = min(len(content), CHUNK_MAX_SIZE)

        chunk_content = content[:end_byte]  # type: ignore
        _write_chunk(chunk_content, storage, chunk_names, key)

        content = content[end_byte:]

    index_meta.add_entry(
        chunk_names=chunk_names,
        start_byte=start_byte,
        end_byte=end_byte,
        **extra_sample_meta
    )


def _get_last_chunk(
    key: str, storage: StorageProvider, index_meta: IndexMeta
) -> Tuple[str, memoryview]:
    """Retrieves the name and memoryview of bytes for the last chunk that was written to. This is helpful for
    filling previous chunks before creating new ones.

    Args:
        key (str): Key for where the chunks are located in `storage` relative to its root.
        storage (StorageProvider): StorageProvider where the chunks are stored.
        index_meta (IndexMeta): IndexMeta object that is used to find the last chunk.

    Returns:
        str: Name of the last chunk. If the last chunk doesn't exist, returns an empty string.
        memoryview: Content of the last chunk. If the last chunk doesn't exist, returns empty memoryview of bytes.
    """
    if len(index_meta.entries) > 0:
        entry = index_meta.entries[-1]
        last_chunk_name = entry["chunk_names"][-1]
        last_chunk_key = get_chunk_key(key, last_chunk_name)
        last_chunk = memoryview(storage[last_chunk_key])
        return last_chunk_name, last_chunk
    return "", memoryview(bytes())


def _generate_chunk_name() -> str:
    return str(uuid1())


def _min_chunk_ct_for_data_size(size: int) -> int:
    """Calculates the minimum number of chunks in which data of given size can be fit."""
    return ceil(size / CHUNK_MAX_SIZE)


def _chunk_has_space(chunk: memoryview) -> bool:
    """Returns whether the given chunk has space to take in more data."""
    return len(chunk) > 0 and len(chunk) < CHUNK_MIN_TARGET


def _write_chunk(
    content: Union[memoryview, bytearray],
    storage: StorageProvider,
    chunk_names: List[str],
    key: str,
    chunk_name: Optional[str] = None,
):
    chunk_name = chunk_name or _generate_chunk_name()
    chunk_names.append(chunk_name)
    chunk_key = get_chunk_key(key, chunk_name)
    storage[chunk_key] = content
