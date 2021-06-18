from hub.constants import UNCOMPRESSED
from hub.util.load import Sample
from hub.core.meta.tensor_meta import TensorMeta
import numpy as np
from hub.core.meta.index_meta import IndexMeta
from typing import Sequence, Tuple, Dict
from uuid import uuid1

from hub.core.typing import StorageProvider
from hub.util.keys import get_chunk_key

from .chunker import generate_chunks


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
    """Write a list of `Sample`s to `storage` under `key`. Updates `tensor_meta` and `index_meta` as needed."""

    for sample in samples:
        if not isinstance(sample, Sample):
            raise Exception()  # TODO

        tensor_meta.check_array_sample_is_compatible(sample.array)

        if sample.compression == UNCOMPRESSED:
            buffer = sample.uncompressed_bytes()
        else:
            buffer = sample.compressed_bytes()

        write_bytes(
            memoryview(buffer),
            key,
            storage,
            tensor_meta,
            index_meta=index_meta,
            extra_sample_meta={  # TODO: convert to kwargs
                "shape": sample.shape,
                "compression": sample.compression,
                "dtype": sample.dtype,
            },
        )

        tensor_meta.update_with_sample(sample.array)
        tensor_meta.length += 1


def write_bytes(
    b: memoryview,
    key: str,
    storage: StorageProvider,
    tensor_meta: TensorMeta,
    index_meta: IndexMeta,
    extra_sample_meta: dict = {},
):
    """Chunk and write bytes to storage, also updates `index_meta`/`tensor_meta`. The provided bytes are treated as a single sample.

    Args:
        b (memoryview): Bytes (as memoryview) to be chunked/written. `b` is considered to be 1 sample and will be
            chunked according to `chunk_size`.
        key (str): Key for where the index_meta and tensor_meta are located in `storage` relative to it's root.
            A subdirectory is created under this `key` (defined in `constants.py`), which is where the chunks will be
            stored.
        storage (StorageProvider): StorageProvider for storing the chunks, index_meta, and tensor_meta.
        tensor_meta (TensorMeta): TensorMeta object that will be written to.
        index_meta (IndexMeta): IndexMeta object that will be written to.
        extra_sample_meta (dict): By default `chunk_names`, `start_byte`, and `end_byte` are written, however
            `IndexMeta.add_entry` supports more parameters than this. Anything passed in this dict will also be used
            to call `IndexMeta.add_entry`.
    """

    if len(b) <= 0:
        write_empty_sample(index_meta, extra_sample_meta)

    # TODO: `_get_last_chunk(...)` is called during an inner loop. memoization here OR having an argument is preferred
    #  for performance
    last_chunk_name, last_chunk = _get_last_chunk(key, storage, index_meta)

    # refactor TODO: move to separate function
    bytes_left_in_last_chunk = 0
    extend_last_chunk = False
    if len(index_meta.entries) > 0 and len(last_chunk) < tensor_meta.chunk_size:
        bytes_left_in_last_chunk = tensor_meta.chunk_size - len(last_chunk)
        last_chunk = bytearray(last_chunk)  # type: ignore
        extend_last_chunk = True

    chunk_generator = generate_chunks(
        b, tensor_meta.chunk_size, bytes_left_in_last_chunk=bytes_left_in_last_chunk
    )

    # refactor TODO: move to separate function
    chunk_names = []
    start_byte = 0
    for chunk in chunk_generator:
        if extend_last_chunk:
            chunk_name = last_chunk_name

            last_chunk += chunk  # type: ignore
            chunk = memoryview(last_chunk)

            start_byte = index_meta.entries[-1]["end_byte"]

            if len(chunk) >= tensor_meta.chunk_size:
                extend_last_chunk = False
        else:
            chunk_name = _random_chunk_name()

        end_byte = len(chunk)

        chunk_key = get_chunk_key(key, chunk_name)
        storage[chunk_key] = chunk

        chunk_names.append(chunk_name)

        last_chunk = memoryview(chunk)
        last_chunk_name = chunk_name

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


def _random_chunk_name() -> str:
    return str(uuid1())
