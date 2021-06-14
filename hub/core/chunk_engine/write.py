from hub.constants import MB
from hub.core.meta.index_meta import IndexMeta
from typing import List, Tuple
from uuid import uuid1

from hub.core.typing import StorageProvider
from hub.util.keys import get_chunk_key
from math import ceil


CHUNK_MAX_SIZE = 32 * MB  # chunks won't ever be bigger than this
CHUNK_MIN_TARGET = 16 * MB  # some chunks might be smaller than this


def write_bytes(
    content: memoryview,
    key: str,
    chunk_size: int,
    storage: StorageProvider,
    index_meta: IndexMeta,
) -> dict:
    """Chunk and write bytes to storage and return the index_meta entry. The provided bytes are treated as a single
        sample.

    Args:
        content (memoryview): Bytes (as memoryview) to be chunked/written. `b` is considered to be 1 sample and will be
            chunked according to `chunk_size`.
        key (str): Key for where the index_meta, and tensor_meta are located in `storage` relative to it's root.
            A subdirectory is created under this `key` (defined in `constants.py`), which is where the chunks will be
            stored.
        chunk_size (int): Desired length of each chunk.
        storage (StorageProvider): StorageProvider for storing the chunks, index_meta, and tensor_meta.
        index_meta (list): List of dictionaries that represent each sample. An entry for `index_meta` is returned
            but not appended to `index_meta`.

    Returns:
        dict: Index map entry (note: it does not get appended to the `index_meta` argument). Dictionary keys:
            chunk_names: Sequential list of names of chunks that were created.
            start_byte: Start byte for this sample.
            end_byte: End byte for this sample. Will be equal to the length of the last chunk written to.
    """  # TODO: Update docstring
    last_chunk_name, last_chunk = _get_last_chunk(key, storage, index_meta)
    start_byte = 0
    chunk_names = []
    if len(last_chunk) > 0:  # last chunk exists
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
    return {"chunk_names": chunk_names, "start_byte": start_byte, "end_byte": end_byte}


def _get_last_chunk(
    key: str, storage: StorageProvider, index_meta: IndexMeta
) -> Tuple[str, memoryview]:
    """Retrieves the name and memoryview of bytes for the last chunk that was written to. This is helpful for
    filling previous chunks before creating new ones.

    Args:
        key (str): Key for where the chunks are located in `storage` relative to it's root.
        storage (StorageProvider): StorageProvider where the chunks are stored.

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
    return ceil(len(size) / CHUNK_MAX_SIZE)
