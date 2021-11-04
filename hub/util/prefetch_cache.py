from functools import lru_cache
from typing import Dict, Tuple, Union, List
from hub.core.storage import S3Provider, StorageProvider, SharedMemoryProvider
from hub.util.keys import get_chunk_key
from hub.util.shared_memory import remove_shared_memory_from_resource_tracker


@lru_cache()
def get_s3_storage(state: tuple) -> S3Provider:
    """Ensures that s3 clients aren't initialized over and over again in the same process"""
    s3 = S3Provider.__new__(S3Provider)
    s3.__setstate__(state)
    return s3


def read_and_store_chunk_group(
    chunk_group: List[Tuple[str, str]],
    shared_memory_names: List[str],
    storage: Union[StorageProvider, tuple],
    commit_id: str,
):
    """Reads chunks from the dataset's storage provider and stores them in the SharedMemory"""
    # TODO: modify to support chunk-wise decompression
    # TODO: if there's sample compression, then we need to decompress each sample present in the chunks before sending to SharedMemory to reduce work on the SharedMemory
    remove_shared_memory_from_resource_tracker()
    if isinstance(storage, tuple):
        state: tuple = storage
        storage = get_s3_storage(state)

    chunk_sizes: Dict[str, int] = {}
    for (key, chunk_name), shared_memory_name in zip(chunk_group, shared_memory_names):
        chunk_key = get_chunk_key(key, chunk_name, commit_id)
        chunk_bytes = storage[chunk_key]
        chunk_size = len(chunk_bytes)
        chunk_sizes[shared_memory_name] = chunk_size
        shared_memory = SharedMemoryProvider()
        shared_memory[shared_memory_name] = chunk_bytes
    return chunk_sizes
