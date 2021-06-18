from hub.constants import (
    CHUNK_MAX_SIZE,
    UNCOMPRESSED,
    USE_UNIFORM_COMPRESSION_PER_SAMPLE,
)
from hub.core.sample import Sample
from hub.core.meta.tensor_meta import TensorMeta
import numpy as np
from hub.core.meta.index_meta import IndexMeta
from typing import List, Optional, Sequence, Tuple, Dict, Union
from uuid import uuid1

from hub.core.typing import StorageProvider
from hub.util.keys import get_chunk_key
from math import ceil


def write_empty_sample(index_meta, extra_sample_meta: dict = {}):
    """Simply adds an entry to `index_meta` that symbolizes an empty array."""

    index_meta.add_entry(chunk_names=[], start_byte=0, end_byte=0, **extra_sample_meta)


def write_samples(
    samples: Sequence[Sample],
    key: str,
    storage: StorageProvider,
    tensor_meta: TensorMeta,
    index_meta: IndexMeta,
):
    """Write a sequence of `Sample`s to `storage` under `key`. Updates `tensor_meta` and `index_meta` as needed.
    This is also where sample-wise compression is handled.
    """

    for sample in samples:
        tensor_meta.check_array_sample_is_compatible(sample.array)

        extra_sample_meta = {  # TODO: convert to kwargs
            "shape": sample.shape,
        }

        if sample.is_empty:
            # if sample is empty, `sample.compression` will always be `UNCOMPRESSED`
            write_empty_sample(index_meta, extra_sample_meta=extra_sample_meta)
        else:

            # TODO: minify this
            if USE_UNIFORM_COMPRESSION_PER_SAMPLE:
                compression = tensor_meta.sample_compression
            else:
                if tensor_meta.sample_compression == UNCOMPRESSED:
                    compression = UNCOMPRESSED
                else:
                    compression = sample.compression
                    if compression == UNCOMPRESSED:
                        compression = tensor_meta.sample_compression

            buffer = sample.compressed_bytes(compression)

            write_bytes(
                memoryview(buffer),
                key,
                storage,
                tensor_meta,
                index_meta=index_meta,
                extra_sample_meta=extra_sample_meta,
            )

        tensor_meta.update_with_sample(sample.array)
        tensor_meta.length += 1


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
        content (memoryview): Bytes (as memoryview) to be chunked/written. considered to be 1 sample and will be
            chunked according to `tensor_meta.chunk_size` and  `CHUNK_MAX_SIZE`.
        key (str): Key for where the index_meta, and tensor_meta are located in `storage` relative to its root.
            A subdirectory is created under this `key` (defined in `constants.py`), which is where the chunks will be
            stored.
        storage (StorageProvider): StorageProvider for storing the chunks, index_meta, and tensor_meta.
        tensor_meta (TensorMeta): TensorMeta object that will be written to.
        index_meta (IndexMeta): IndexMeta object that will be written to.
        extra_sample_meta (dict): By default `chunk_names`, `start_byte`, and `end_byte` are written, however
            `IndexMeta.add_entry` supports more parameters than this. Anything passed in this dict will also be used
            to call `IndexMeta.add_entry`.

    Raises:
        ValueError: `b` shouldn't be empty.
    """

    if len(content) <= 0:
        raise ValueError(
            "Empty samples should not be written via `write_bytes`. Please use `write_empty_sample`."
        )

    # TODO: `_get_last_chunk(...)` is called during an inner loop. memoization here OR having an argument is preferred
    #  for performance

    # TODO pass CHUNK_MAX and read from tensor_meta instead of using constants
    last_chunk_name, last_chunk = _get_last_chunk(key, storage, index_meta)
    start_byte = 0
    chunk_names: List[str] = []

    if _chunk_has_space(last_chunk, tensor_meta.chunk_size):
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
        **extra_sample_meta,
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

    for entry in reversed(index_meta.entries):
        chunk_names = entry["chunk_names"]
        if len(chunk_names) > 0:
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


def _chunk_has_space(chunk: memoryview, chunk_min_target: int) -> bool:
    """Returns whether the given chunk has space to take in more data."""
    return len(chunk) > 0 and len(chunk) < chunk_min_target


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
