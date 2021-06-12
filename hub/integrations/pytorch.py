from hub.util.remove_cache import remove_memory_cache
from hub.util.join_chunks import join_chunks
from hub.core.meta.tensor_meta import read_tensor_meta
import os
import numpy as np
from itertools import repeat
from collections import defaultdict
from typing import Any, Callable, List, Optional, Set, Dict
from hub.core.meta.index_map import read_index_map
from hub.util.exceptions import ModuleNotInstalledException
from hub.util.shared_memory import (
    remove_shared_memory_from_resource_tracker,
    clear_shared_memory,
)
from pathos.pools import ProcessPool  # type: ignore
from hub.core.storage import MemoryProvider

try:
    from multiprocessing.shared_memory import SharedMemory  # type: ignore
except ModuleNotFoundError:
    pass

_hub_storage_provider = MemoryProvider()

# TODO make this use shared memory to make on the fly transforms faster. Currently using transform slows us down by 10x
def _apply_transform(transform: Callable, sample: Dict):
    """Used to apply transforms to a single sample"""
    return transform(sample) if transform else sample


def _read_and_store_chunk(chunk_name: str, shared_memory_name: str, key: str):
    """Reads a single chunk from the dataset's storage provider and stores it in the SharedMemory. Returns its size"""
    remove_shared_memory_from_resource_tracker()
    chunk_path = os.path.join(key, "chunks", chunk_name)
    chunk_bytes = _hub_storage_provider[chunk_path]
    chunk_size = len(chunk_bytes)
    shm = SharedMemory(create=True, size=chunk_size, name=shared_memory_name)

    # needs to be done as some OS allocate extra space
    shm.buf[0:chunk_size] = chunk_bytes
    shm.close()
    return chunk_size


def dataset_to_pytorch(dataset, transform: Callable = None, workers: int = 1):
    return TorchDataset(dataset, transform, workers)


class TorchDataset:
    def __init__(self, dataset, transform: Callable = None, workers: int = 1):
        self.dataset = dataset
        self._set_globals()
        self.transform: Optional[Callable] = transform
        self.workers: int = workers
        self.map = ProcessPool(nodes=workers).map
        self.len = len(dataset)
        self.keys = list(self.dataset.tensors)

        # contains meta for each Tensor
        self.all_meta: Dict[str, Dict] = self._load_all_meta()

        # contains index_map for each Tensor
        self.all_index_maps: Dict[str, List] = self._load_all_index_maps()

        # stores index-value map for each Tensor where value is the actual array at the index
        # acts as in memory prefetch cache
        self.all_index_value_maps: Dict[str, Dict[int, Any]] = defaultdict(dict)

        # tracks last index that was prefetched in the prefetch cache for each Tensor
        self.last_index_map: Dict[str, int] = {}

        # in memory processed cache containing all samples generated after prefetching and transforming
        self.processed_samples: List[Dict] = []
        self.processed_range = slice(-1, -1)  # range of processed_samples

        # keeps track of names of all shared_memory that have data in them
        self.all_shared_memory_names: Dict[str, List[str]] = defaultdict(list)

        # keeps pointers to shared memory across tensors so they don't get closed between calls to getitem
        self.all_shared_memory: Dict = defaultdict(list)

        self.last_chunk_num_generated = -1

    def __len__(self):
        return self.len

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
    def _set_globals(self):
        """Sets the global values for storage provider and a few plugins"""
        global torch
        try:
            import torch
        except ModuleNotFoundError:
            raise ModuleNotInstalledException(
                "'torch' should be installed to convert the Dataset into pytorch format"
            )

        # global to pass to processes, not possible to serialize and send
        global _hub_storage_provider

        # TODO boto3.client isn't safe for multiprocessing https://github.com/boto/boto3/pull/2848/files
        # could it be working here as we're only reading data?
        _hub_storage_provider = remove_memory_cache(self.dataset.storage)

    def _load_all_index_maps(self):
        """Loads index maps for all Tensors into memory"""
        all_index_maps = {
            key: read_index_map(key, _hub_storage_provider) for key in self.keys
        }
        return all_index_maps

    def _load_all_meta(self):
        """Loads meta for all Tensors into memory"""
        all_meta = {}
        # pytorch doesn't support certain dtypes, which are type casted to another dtype implicitly
        for key in self.keys:
            meta = read_tensor_meta(key, _hub_storage_provider)
            if meta["dtype"] == "uint16":
                meta["dtype"] = "int32"
            elif meta["dtype"] in ["uint32", "uint64"]:
                meta["dtype"] = "int64"
            all_meta[key] = meta
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
        chunk_sizes: List[int] = self.map(
            _read_and_store_chunk, chunk_names, shared_memory_names, repeat(key)
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
        index_map = self.all_index_maps[key]
        while len(chunk_names) < self.workers and index < len(self):
            chunks = index_map[index]["chunk_names"]
            chunk_names.update(chunks)
            index += 1
        return chunk_names

    def _np_from_chunk_list(self, index: int, key: str, chunks: List[bytes]):
        """Takes a list of chunks and returns a numpy array from it"""
        index_entry = self.all_index_maps[key][index]

        start_byte = index_entry["start_byte"]
        end_byte = index_entry["end_byte"]
        dtype = self.all_meta[key]["dtype"]
        shape = index_entry["shape"]

        combined_bytes = join_chunks(chunks, start_byte, end_byte)
        if isinstance(combined_bytes, memoryview):
            arr = np.frombuffer(combined_bytes, dtype=dtype).reshape(shape)
            combined_bytes.release()
        else:
            arr = np.frombuffer(combined_bytes, dtype=dtype).reshape(shape)
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
            index_entry = self.all_index_maps[key][i]
            for chunk_name in index_entry["chunk_names"]:
                if chunk_name not in chunk_map:
                    self.last_index_map[key] = i - 1
                    return
                chunks.append(chunk_map[chunk_name])
            self.all_index_value_maps[key][i] = self._np_from_chunk_list(i, key, chunks)

        self.last_index_map[key] = len(self) - 1

    def _process_samples(self):
        """Processes the prefetched values from across tensors into dictionaries.
        These samples may be further processed if a transform is specified.
        """
        first_index = self.processed_range.stop + 1
        # different no. of samples are fetched for each tensor, take the min and process
        last_index = min(self.last_index_map[key] for key in self.keys)
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
