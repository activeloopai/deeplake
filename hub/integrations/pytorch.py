from hub.core.storage import StorageProvider, S3Provider, MemoryProvider
from hub.core.compression import decompress_array
from hub.constants import UNCOMPRESSED
from hub.core.meta.index_meta import IndexMeta
from hub.core.meta.tensor_meta import TensorMeta
from hub.util.remove_cache import remove_memory_cache
from hub.util.join_chunks import join_chunks
import os
import numpy as np
from itertools import repeat
from collections import defaultdict
from typing import Any, Callable, List, Optional, Set, Dict, Union
from hub.util.exceptions import DatasetUnsupportedPytorch, ModuleNotInstalledException
from hub.util.shared_memory import (
    remove_shared_memory_from_resource_tracker,
    clear_shared_memory,
)
from pathos.pools import ProcessPool  # type: ignore

try:
    from multiprocessing.shared_memory import SharedMemory  # type: ignore
except ModuleNotFoundError:
    pass

from functools import lru_cache


@lru_cache()
def get_s3_storage(state: tuple) -> S3Provider:
    """Ensures that s3 clients aren't initialized over and over again in the same process"""
    s3 = S3Provider.__new__(S3Provider)
    s3.__setstate__(state)
    return s3


# TODO make this use shared memory to make on the fly transforms faster. Currently using transform slows us down by 10x
def _apply_transform(transform: Callable, sample: Dict):
    """Used to apply transforms to a single sample"""
    return transform(sample) if transform else sample


def _read_and_store_chunk(
    chunk_name: str,
    shared_memory_name: str,
    key: str,
    storage: Union[StorageProvider, tuple],
):
    """Reads a single chunk from the dataset's storage provider and stores it in the SharedMemory. Returns its size"""

    # TODO: modify to support chunk-wise decompression
    remove_shared_memory_from_resource_tracker()
    if isinstance(storage, tuple):
        state: tuple = storage
        storage = get_s3_storage(state)
    chunk_path = os.path.join(key, "chunks", chunk_name)
    chunk_bytes = storage[chunk_path]
    chunk_size = len(chunk_bytes)
    shm = SharedMemory(create=True, size=chunk_size, name=shared_memory_name)

    # needs to be done as some OS (like macOS) allocate extra space
    shm.buf[0:chunk_size] = chunk_bytes
    shm.close()
    return chunk_size


def dataset_to_pytorch(dataset, transform: Callable = None, workers: int = 1):
    dataset.flush()
    return TorchDataset(dataset, transform, workers)


class TorchDataset:
    def __init__(self, dataset, transform: Callable = None, workers: int = 1):
        self._import_torch()
        self.transform: Optional[Callable] = transform
        self.workers: int = workers
        self.map = ProcessPool(nodes=workers).map
        self.length = len(dataset)
        self.keys = list(dataset.tensors)
        self.storage = remove_memory_cache(dataset.storage)

        if isinstance(self.storage, MemoryProvider):
            raise DatasetUnsupportedPytorch(
                "Datasets whose underlying storage is MemoryProvider are not supported for Pytorch iteration."
            )

        elif isinstance(self.storage, S3Provider):
            self.storage_state_tuple = self.storage.__getstate__()

        # contains meta for each Tensor
        self.all_tensor_metas: Dict[str, TensorMeta] = self._load_all_meta()
        index_value = dataset.index.values[0].value

        if not isinstance(index_value, slice):
            raise DatasetUnsupportedPytorch(
                "Only full dataset or dataset indexed using slices can be converted to pytorch."
            )

        if index_value.step not in [None, 1]:
            raise DatasetUnsupportedPytorch(
                "The step of the Dataset object is not None or 1"
            )

        self.index_offset = index_value.start or 0

        # contains index_meta for each Tensor
        self.all_index_metas: Dict[str, IndexMeta] = self._load_all_index_meta()

        # stores index-value map for each Tensor where value is the actual array at the index
        # acts as in memory prefetch cache
        self.all_index_value_maps: Dict[str, Dict[int, Any]] = defaultdict(dict)

        # tracks last index that was prefetched in the prefetch cache for each Tensor
        self.last_index_meta: Dict[str, int] = {}

        # in memory processed cache containing all samples generated after prefetching and transforming
        self.processed_samples: List[Dict] = []
        self.processed_range = slice(-1, -1)  # range of processed_samples

        # keeps track of names of all shared_memory that have data in them
        self.all_shared_memory_names: Dict[str, List[str]] = defaultdict(list)

        # keeps pointers to shared memory across tensors so they don't get closed between calls to getitem
        self.all_shared_memory: Dict = defaultdict(list)

        self.last_chunk_num_generated = -1

    def __len__(self):
        return self.length

    def __getitem__(self, index: int):
        for key in self.keys:
            # prefetch cache miss, fetch data
            if index not in self.all_index_value_maps[key]:
                self._prefetch_data(key, index)
        # processed cache miss, process more samples
        if index > self.processed_range.stop:
            self._process_samples()
        sample = self.processed_samples[index - self.processed_range.start]

        if index == len(self) - 1:  # clean up at the end
            self._all_shared_memory_clean_up()
            self.processed_range = slice(-1, -1)

        return sample

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]

    # helper functions
    def _import_torch(self):
        global torch
        try:
            import torch
        except ModuleNotFoundError:
            raise ModuleNotInstalledException(
                "'torch' should be installed to convert the Dataset into pytorch format"
            )

    def _load_all_index_meta(self):
        """Loads index metas for all Tensors into memory"""
        all_index_metas = {key: IndexMeta.load(key, self.storage) for key in self.keys}
        return all_index_metas

    def _load_all_meta(self):
        """Loads meta for all Tensors into memory"""
        all_meta = {}
        # pytorch doesn't support certain dtypes, which are type casted to another dtype implicitly
        for key in self.keys:
            tensor_meta = TensorMeta.load(key, self.storage)
            if tensor_meta.dtype == "uint16":
                tensor_meta.dtype = "int32"
            elif tensor_meta.dtype in ["uint32", "uint64"]:
                tensor_meta.dtype = "int64"
            all_meta[key] = tensor_meta
        return all_meta

    def _prefetch_data(self, key: str, index: int):
        """Prefetches data for the given key, starting from the given index"""
        # clear data from previous prefetching, before fetching data
        del self.all_index_value_maps[key]
        old_shared_memory_names = self.all_shared_memory_names[key]
        clear_shared_memory(old_shared_memory_names)
        chunk_names = list(self._get_chunk_names(index, key))
        shared_memory_names = self._generate_shared_memory_names(chunk_names)
        clear_shared_memory(shared_memory_names)

        # will be passing in storage provider to each process
        storage: Union[S3Provider, Dict] = self.storage
        # s3 provider is not sent as storage provider but instead sent as a tuple containing it's state
        if isinstance(storage, S3Provider):
            storage = self.storage_state_tuple

        chunk_sizes: List[int] = self.map(
            _read_and_store_chunk,
            chunk_names,
            shared_memory_names,
            repeat(key),
            repeat(storage),
        )
        self._get_data_from_chunks(
            index, key, chunk_names, shared_memory_names, chunk_sizes
        )
        self.all_shared_memory_names[key] = shared_memory_names

    def _generate_shared_memory_names(self, chunk_names: List[str]):
        """Generates a name for every chunk_name as chunknames very large and fail on MacOS"""
        ls = []
        for _ in chunk_names:
            self.last_chunk_num_generated += 1
            ls.append(f"al_{self.last_chunk_num_generated}")
        return ls

    def _get_chunk_names(self, index: int, key: str):
        """Gets chunk names for elements starting from index to read in parallel"""
        chunk_names: Set[str] = set()
        index_meta = self.all_index_metas[key]
        while len(chunk_names) < self.workers and index < len(self):
            actual_index = self.index_offset + index
            chunks = index_meta.entries[actual_index]["chunk_names"]
            chunk_names.update(chunks)
            index += 1
        return chunk_names

    def _np_from_chunk_list(self, index: int, key: str, chunks: List[bytes]):
        """Takes a list of chunks and returns a numpy array from it"""

        # TODO: this function should be located in core (sample_from_index_entry doesn't work because prefetch cache)
        # TODO: i think this can be done by creating a `PrefetchCache` like how we have `LRUCache` then all of this code
        # TODO: will be hanlded in there

        index_entry = self.all_index_metas[key].entries[index]

        start_byte = index_entry["start_byte"]
        end_byte = index_entry["end_byte"]
        shape = index_entry["shape"]

        tensor_meta = self.all_tensor_metas[key]
        dtype = tensor_meta.dtype
        sample_compression = tensor_meta.sample_compression

        combined_bytes = join_chunks(chunks, start_byte, end_byte)

        # TODO: migrate UNCOMPRESSED check into a single function
        if sample_compression == UNCOMPRESSED:
            arr = np.frombuffer(combined_bytes, dtype=dtype).reshape(shape)
        else:
            arr = decompress_array(combined_bytes)

        if isinstance(combined_bytes, memoryview):
            combined_bytes.release()

        return arr

    def _get_data_from_chunks(
        self,
        index: int,
        key: str,
        chunk_names: List[str],
        shared_memory_names: List[str],
        chunk_sizes: List[int],
    ):
        """Extracts data from all the chunks in chunk_names and stores it index wise in a dictionary"""
        self.all_index_value_maps[key] = {}
        chunk_map = {}
        # loads value of chunks saved in SharedMemory into memory
        for chunk_name, shared_memory_name, chunk_size in zip(
            chunk_names, shared_memory_names, chunk_sizes
        ):
            self.all_shared_memory[key].append(SharedMemory(name=shared_memory_name))
            chunk_map[chunk_name] = self.all_shared_memory[key][-1].buf[:chunk_size]

        # saves np array for each index in memory
        for i in range(index, len(self)):
            chunks = []
            actual_index = self.index_offset + i
            index_entry = self.all_index_metas[key].entries[actual_index]
            for chunk_name in index_entry["chunk_names"]:
                if chunk_name not in chunk_map:
                    self.last_index_meta[key] = i - 1
                    return
                chunks.append(chunk_map[chunk_name])
            self.all_index_value_maps[key][i] = self._np_from_chunk_list(
                actual_index, key, chunks
            )

        self.last_index_meta[key] = len(self) - 1

    def _process_samples(self):
        """Processes the prefetched values from across tensors into dictionaries.
        These samples may be further processed if a transform is specified.
        """
        first_index = self.processed_range.stop + 1
        # different no. of samples are fetched for each tensor, take the min and process
        last_index = min(self.last_index_meta[key] for key in self.keys)
        samples = []
        for i in range(first_index, last_index + 1):
            sample = {key: self.all_index_value_maps[key][i] for key in self.keys}
            samples.append(sample)

        if self.transform:
            self.processed_samples = self.map(
                _apply_transform, repeat(self.transform), samples
            )
        else:
            self.processed_samples = samples
        self.processed_range = slice(first_index, last_index)

    def _all_shared_memory_clean_up(self):
        """Cleans up possibly leaked memory at the end of iteration across Tensors"""
        for key in self.keys:
            shared_memory_names = self.all_shared_memory_names[key]
            clear_shared_memory(shared_memory_names)
