import os
import pickle
import numpy as np

from .chunker import join_chunks

from hub.util.keys import get_meta_key, get_index_map_key

from hub.core.typing import Provider
from typing import Callable, List, Union


def read_array(
    key: str,
    storage: Provider,
    array_slice: slice = slice(None),
) -> np.ndarray:
    """Read and join chunks into an array from storage.

    Args:
        key (str): Key for where the chunks, index_map, and meta are located in `storage` relative to it's root.
        array_slice (slice): Slice that represents which samples to read. Default = slice representing all samples.
        storage (Provider): Provider for reading the chunks, index_map, and meta.

    Returns:
        np.ndarray: Array containing the sample(s) in the `array_slice` slice.
    """

    # TODO: don't use pickle
    meta = pickle.loads(storage[get_meta_key(key)])
    index_map = pickle.loads(storage[get_index_map_key(key)])

    # TODO: read samples in parallel
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
