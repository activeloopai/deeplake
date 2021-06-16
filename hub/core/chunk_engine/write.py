from hub.constants import CHUNK_MAX_SIZE, CHUNK_MIN_TARGET
from hub.core.meta.tensor_meta import TensorMeta
import numpy as np
from hub.core.meta.index_meta import IndexMeta
from typing import List, Tuple
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
        content (memoryview): Bytes (as memoryview) to be chunked/written. `b` is considered to be 1 sample and will be
            chunked according to `chunk_size`.
        key (str): Key for where the index_meta, and tensor_meta are located in `storage` relative to it's root.
            A subdirectory is created under this `key` (defined in `constants.py`), which is where the chunks will be
            stored.
        storage (StorageProvider): StorageProvider for storing the chunks, index_meta, and tensor_meta.
        tensor_meta (TensorMeta): TensorMeta object that will be written to.
        index_meta (IndexMeta): IndexMeta object that will be written to.
        extra_sample_meta (dict): By default `chunk_names`, `start_byte`, and `end_byte` are written, however
            `IndexMeta.add_entry` supports more parameters than this. Anything passed in this dict will also be used
            to call `IndexMeta.add_entry`.
    """
    # TODO pass CHUNK_MIN, CHUNK_MAX instead of using constants
    # do we need min target?

    last_chunk_name, last_chunk = _get_last_chunk(key, storage, index_meta)
    start_byte = 0
    chunk_names = []

    if (
        len(last_chunk) > 0 and len(last_chunk) < CHUNK_MIN_TARGET
    ):  # last chunk exists and has space
        last_chunk_size = len(last_chunk)
        num_chunks_b = _get_chunk_count(len(content))
        extra_bytes_in_last_chunk = min(len(content), CHUNK_MAX_SIZE - last_chunk_size)
        num_chunks_after_combining = _get_chunk_count(len(content) + last_chunk_size)
        if num_chunks_after_combining == num_chunks_b:  # combine if count is same
            start_byte = index_meta.entries[-1]["end_byte"]
            chunk_names.append(last_chunk_name)
            last_chunk = bytearray(last_chunk) + content
            chunk_key = get_chunk_key(key, last_chunk_name)
            storage[chunk_key] = last_chunk
            end_byte = len(last_chunk)
            content = content[extra_bytes_in_last_chunk:]

    while len(content) > 0:
        chunk_name = _generate_chunk_name()
        chunk_names.append(chunk_name)
        chunk_key = get_chunk_key(key, chunk_name)
        chunk_size = min(len(content), CHUNK_MAX_SIZE)
        storage[chunk_key] = content[0:chunk_size]
        end_byte = chunk_size
        content = content[chunk_size:]

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
        key (str): Key for where the chunks are located in `storage` relative to it's root.
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


def _get_chunk_count(size):
    """Returns the minimum number of chunks in which data of given size can be fit."""
    return ceil(size / CHUNK_MAX_SIZE)
