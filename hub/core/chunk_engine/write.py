from typing import List, Tuple
from uuid import uuid1

from hub.core.typing import StorageProvider
from hub.util.keys import get_chunk_key
from .chunker import generate_chunks


def write_bytes(
    b: memoryview,
    key: str,
    chunk_size: int,
    storage: StorageProvider,
    index_map: List[dict],
) -> dict:
    """Chunk and write bytes to storage and return the index_map entry. The provided bytes are treated as a single
        sample.

    Args:
        b (memoryview): Bytes (as memoryview) to be chunked/written. `b` is considered to be 1 sample and will be
            chunked according
            to `chunk_size`.
        key (str): Key for where the index_map, and tensor_meta are located in `storage` relative to it's root.
            A subdirectory is created under this `key` (defined in `constants.py`), which is where the chunks will be
            stored.
        chunk_size (int): Desired length of each chunk.
        storage (StorageProvider): StorageProvider for storing the chunks, index_map, and tensor_meta.
        index_map (list): List of dictionaries that represent each sample. An entry for `index_map` is returned
            but not appended to `index_map`.

    Returns:
        dict: Index map entry (note: it does not get appended to the `index_map` argument). Dictionary keys:
            chunk_names: Sequential list of names of chunks that were created.
            start_byte: Start byte for this sample. Will be 0 if no previous chunks exist, otherwise will
                be set to the length of the last chunk before writing.
            end_byte: End byte for this sample. Will be equal to the length of the last chunk written to.
    """

    # TODO: `_get_last_chunk(...)` is called during an inner loop. memoization here OR having an argument is preferred
    #  for performance
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
    index_map_entry = {
        "chunk_names": chunk_names,
        "start_byte": start_byte,
        "end_byte": end_byte,
    }

    return index_map_entry


def _get_last_chunk(
    key: str, index_map: List[dict], storage: StorageProvider
) -> Tuple[str, memoryview]:
    """Retrieves the name and memoryview of bytes for the last chunk that was written to. This is helpful for
    filling previous chunks before creating new ones.

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
