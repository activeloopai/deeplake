import os
import pickle
import numpy as np

from .chunker import join_chunks
from .util import get_meta_key, get_index_map_key

from hub.core.typing import Provider
from typing import Callable, List, Union


def read_array(
    key: str,
    storage: Provider,
    array_slice: slice = slice(None),
) -> np.ndarray:
    """Read & join chunks into an array from storage.

    Args:
        key (str): Key for where the chunks/index_map/meta will be located in `storage` relative to it's root.
        array_slice (slice): Slice that represents which samples to read. Default = returns all samples.
        storage (Provider): Provider for reading the chunks, index_map, & meta.

    Returns:
        np.ndarray: Array containing the sample(s) in the `index` slice.
    """

    # TODO: don't use pickle
    meta = pickle.loads(storage[get_meta_key(key)])
    index_map = pickle.loads(storage[get_index_map_key(key)])

    samples = []
    for index_entry in index_map[array_slice]:
        chunks = []
        for chunk_name in index_entry["chunk_names"]:
            chunk_key = os.path.join(key, "chunks", chunk_name)
            chunk = storage[chunk_key]

            chunks.append(chunk)

        combined_bytes = join_chunks(
            chunks,
            index_entry["start_byte"],
            index_entry["end_byte"],
        )

        out_array = np.frombuffer(combined_bytes, dtype=meta["dtype"])
        samples.append(out_array.reshape(index_entry["shape"]))

    return np.array(samples)
