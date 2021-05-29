import os
import numpy as np
import json
import pickle

from .chunker import join_chunks

from hub import constants
from hub.util.keys import get_meta_key, get_index_map_key

from hub.core.typing import StorageProvider
from typing import Callable, List, Union


def read_tensor_meta(key: str, storage: StorageProvider):
    return json.loads(storage[get_meta_key(key)])


def read_index_map(key: str, storage: StorageProvider):
    return json.loads(storage[get_index_map_key(key)])


def read_dataset_meta(storage: StorageProvider):
    return json.loads(storage[constants.META_FILENAME])


def key_exists(key: str, storage: StorageProvider):
    meta_key = get_meta_key(key)
    index_map_key = get_index_map_key(key)
    return meta_key in storage or index_map_key in storage


def read_array(
    key: str,
    storage: StorageProvider,
    array_slice: slice = slice(None),
) -> np.ndarray:
    """Read and join chunks into an array from storage.

    Args:
        key (str): Key for where the chunks, index_map, and meta are located in `storage` relative to it's root.
        array_slice (slice): Slice that represents which samples to read. Default = slice representing all samples.
        storage (StorageProvider): StorageProvider for reading the chunks, index_map, and meta.

    Returns:
        np.ndarray: Array containing the sample(s) in the `array_slice` slice.
    """

    meta = read_tensor_meta(key, storage)
    index_map = read_index_map(key, storage)

    # TODO: read samples in parallel
    samples = []
    print("map")
    print(index_map)
    for index_entry in index_map[array_slice]:
        chunks = []
        print("array")
        print(index_entry)
        for i in range(len(index_entry)):
            # print(index_entry)
            if index_entry[i] == "chunk_names":
                for j in range(len(index_entry[i + 1])):
                    chunk_name = index_entry[i + 1][j]
                    chunk_key = os.path.join(key, "chunks", chunk_name)
                    # print("hello")
                    # print(chunk_name)
                    chunk = storage[chunk_key]

                    # print("there")
                    # for chunk_name in index_entry["chunk_names"]:
                    #     chunk_key = os.path.join(key, "chunks", chunk_name)
                    #     chunk = storage[chunk_key]

                    chunks.append(chunk)
                    # print("----")

                    # print("----")
                    combined_bytes = join_chunks(
                        chunks,
                        index_entry[i + 3],
                        index_entry[i + 5],
                    )
                    # print(combined_bytes)
                    # print(type(index_entry[i + 3]))
                    # print(index_entry[i + 5])
                    out_array = np.frombuffer(combined_bytes, dtype=meta["dtype"])
                    # print("OT ARRAY")
                    # print(type(out_array))
                    # print(type(index_entry[i + 7]))
                    samples.append(out_array.reshape(tuple(index_entry[i + 7])))
                    # print(tuple(index_entry[i + 7]))
    # print("eh")

    # print(samples)
    # print("eeh")
    return np.array(samples)

    # # TODO: read samples in parallel
    # samples = []
    # for index_entry in index_map[array_slice]:
    #     chunks = []
    #     for chunk_name in index_entry["chunk_names"]:
    #         chunk_key = os.path.join(key, "chunks", chunk_name)
    #         print("hello")
    #         print(chunk_name)
    #         chunk = storage[chunk_key]
    #         print("there")

    #         chunks.append(chunk)

    #     combined_bytes = join_chunks(
    #         chunks,
    #         index_entry["start_byte"],
    #         index_entry["end_byte"],
    #     )
    #     print(index_entry)
    #     print(index_entry["start_byte"])
    #     print(index_entry["end_byte"])
    #     out_array = np.frombuffer(combined_bytes, dtype=meta["dtype"])
    #     print("OT ARRAY")
    #     print(type(out_array))
    #     samples.append(out_array.reshape(index_entry["shape"]))
    #     print(index_entry["shape"])
    # print("eh")
    # print(samples)
    # print("eeh")
    # return np.array(samples)
