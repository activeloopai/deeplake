import os
import pickle
import numpy as np
from itertools import repeat
from collections import defaultdict
from typing import Callable, List, Set, Dict
from hub.util.keys import get_index_map_key
from hub.core.chunk_engine.chunker import join_chunks
from hub.core.chunk_engine.read import read_tensor_meta
from hub.util.exceptions import ModuleNotInstalledException, RequiresHigherPythonVersion

# TODO make this use shared memory to make on the fly transforms faster. Currently using transform slows us down by 10x
def _apply_transform(transform: Callable, sample: Dict):
    """Used to apply transforms to a single sample"""
    return transform(sample) if transform else sample


def _shared_memory_clear(chunk_names: Set[str]):
    """Checks if an existing SharedMemory exists for any chunk in chunk_names and clears it"""
    for chunk_name in chunk_names:
        try:
            shm = SharedMemory(name=chunk_name)
            shm.close()
            shm.unlink()
        except FileNotFoundError:
            pass


def _read_and_store_chunk(chunk_name: str, key: str):
    """Reads a single chunk from the dataset's storage provider and stores it in the SharedMemory"""
    remove_shared_memory_from_resource_tracker()
    chunk_path = os.path.join(key, "chunks", chunk_name)
    chunk_bytes = _hub_storage_provider[chunk_path]
    chunk_size = len(chunk_bytes)
    shm = SharedMemory(create=True, size=chunk_size, name=chunk_name)
    shm.buf[:] = chunk_bytes
    shm.close()
    return


def remove_shared_memory_from_resource_tracker():
    """Monkey-patch that fixes bug in Python SharedMemory
    More details at: https://bugs.python.org/issue38119
    """

    def fix_register(name, rtype):
        if rtype == "shared_memory":
            return
        return resource_tracker._resource_tracker.register(self, name, rtype)

    def fix_unregister(name, rtype):
        if rtype == "shared_memory":
            return
        return resource_tracker._resource_tracker.unregister(self, name, rtype)

    resource_tracker.register = fix_register
    resource_tracker.unregister = fix_unregister
    if "shared_memory" in resource_tracker._CLEANUP_FUNCS:
        del resource_tracker._CLEANUP_FUNCS["shared_memory"]


def _to_pytorch(dataset, transform: Callable = None, workers: int = 1):
    return TorchDataset(dataset, transform, workers)


class TorchDataset:
    def __init__(self, dataset, transform: Callable = None, workers: int = 1):

        self.dataset = dataset  # TODO disable the memory cache
        self._set_globals()
        self.transform = transform
        self.workers = workers
        self.map = ProcessPool(nodes=workers).map

        # contains meta for each Tensor
        self.all_meta = self._load_meta()

        # contains index_map for each Tensor
        self.all_index_maps = self._load_index_maps()

        # stores index-value map for each Tensor where value is the actual array at the index
        # acts as in memory prefetch cache
        self.all_index_value_maps = defaultdict(dict)

        # tracks last index that was prefetched in the prefetch cache for each Tensor
        self.last_index_map = {}

        # in memory processed cache containing all samples generated after prefetching and transforming
        self.processed_samples = None
        self.processed_range = slice(-1, -1)  # range of processed_samples

        # keeps track of names of all chunks across tensors whose data is currently prefetched
        self.all_chunk_sets = {}

        # keeps pointers to shared memory across tensors so they don't get closed between calls to getitem
        self.all_shared_memory = defaultdict(list)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        for key in self.dataset.tensors:
            # prefetch cache hit
            if index in self.all_index_value_maps[key]:
                continue

            # index exceeded prefetch cache, clear it (and fetch more later)
            if index != 0 and index == self.last_index_map[key] + 1:
                del self.all_index_value_maps[key]
                chunk_names = self.all_chunk_sets[key]
                _shared_memory_clear(chunk_names)

            chunk_names = self._get_chunk_names(index, key)
            _shared_memory_clear(chunk_names)
            self.map(_read_and_store_chunk, chunk_names, repeat(key))
            self._get_data_from_chunks(index, key, chunk_names)
            self.all_chunk_sets[key] = chunk_names

        # if processed cache miss, process more samples
        if index > self.processed_range.stop:
            self._process_samples()
        if index == len(self) - 1:  # clean up at the end
            self._shared_memory_clean_up()
        return self.processed_samples[index - self.processed_range.start]

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]

    # helper functions

    def _set_globals(self):
        """Sets the global values for storage provider and a few plugins"""
        global torch
        global ProcessPool
        global resource_tracker
        global SharedMemory

        try:
            import torch
        except ModuleNotFoundError:
            raise ModuleNotInstalledException("'torch' should be installed to convert the Dataset into pytorch format")

        try:
            from pathos.pools import ProcessPool
        except ModuleNotFoundError:
            raise ModuleNotInstalledException("'pathos' should be installed to convert the Dataset into pytorch format")

        try:
            from multiprocessing import resource_tracker
            from multiprocessing.shared_memory import SharedMemory
        except ImportError:
            raise RequiresHigherPythonVersion("to_pytorch", "3.8")

        global _hub_storage_provider  # global to pass to processes, not possible to serialize and send

        # TODO boto3.client isn't safe for multiprocessing https://github.com/boto/boto3/pull/2848/files
        # could it be working here as we're only reading data?
        _hub_storage_provider = self.dataset.provider

    def _load_index_maps(self):
        """Loads index maps for all Tensors into memory"""
        # TODO there should be an easier way in API to do this
        all_index_maps = {}
        for key in self.dataset.tensors:
            index_map = pickle.loads(_hub_storage_provider[get_index_map_key(key)])
            all_index_maps[key] = index_map
        return all_index_maps

    def _load_meta(self):
        """Loads meta for all Tensors into memory"""
        # TODO there should be an easier way in API to do this
        all_meta = {}
        # pytorch doesn't support certain dtypes, which are type casted to another dtype below
        for key in self.dataset.tensors:
            meta = read_tensor_meta(key, _hub_storage_provider)
            if meta["dtype"] == "uint16":
                meta["dtype"] = "int32"
            elif meta["dtype"] in ["uint32", "uint64"]:
                meta["dtype"] = "int64"
            all_meta[key] = meta
        return all_meta

    def _get_chunk_names(self, index: int, key: str):
        """Gets chunk names for elements starting from index to read in parallel"""
        chunk_names = set()
        index_map = self.all_index_maps[key]
        while len(chunk_names) < self.workers and index < len(self):
            chunks = index_map[index]["chunk_names"]
            chunk_names.update(chunks)
            index += 1
        return chunk_names

    def _np_from_chunk_list(self, index: int, key: str, chunks: List[str]):
        """Takes a list of chunks and returns a numpy array from it"""
        index_entry = self.all_index_maps[key][index]

        start_byte = index_entry["start_byte"]
        end_byte = index_entry["end_byte"]
        dtype = self.all_meta[key]["dtype"]
        shape = index_entry["shape"]

        combined_bytes = join_chunks(chunks, start_byte, end_byte)
        arr = np.frombuffer(combined_bytes, dtype=dtype).reshape(shape)
        combined_bytes.release()
        return arr

    def _get_data_from_chunks(self, index: int, key: str, chunk_names: Set[str]):
        """Extracts data from all the chunks in chunk_set and stores it index wise in a dictionary"""
        self.all_index_value_maps[key] = {}
        chunk_map = {}
        # loads value of chunks saved in SharedMemory into memory
        for chunk_name in chunk_names:
            self.all_shared_memory[key].append(SharedMemory(name=chunk_name))
            chunk_map[chunk_name] = self.all_shared_memory[key][-1].buf[:]

        # saves np array for each index in memory
        for i in range(index, len(self.dataset)):
            chunks = []
            index_entry = self.all_index_maps[key][i]
            for chunk_name in index_entry["chunk_names"]:
                if chunk_name not in chunk_map:
                    self.last_index_map[key] = i - 1
                    return
                chunks.append(chunk_map[chunk_name])
            self.all_index_value_maps[key][i] = self._np_from_chunk_list(i, key, chunks)

        self.last_index_map[key] = len(self.dataset) - 1

    def _process_samples(self):
        """Processes the prefetched values from across tensors into dictionaries.
        These samples may be further processed if a transform is specified.
        """
        first_index = self.processed_range.stop + 1
        # different no. of samples are fetched for each tensor, take the min and process
        last_index = min(self.last_index_map[key] for key in self.dataset.tensors)
        samples = []
        for i in range(first_index, last_index + 1):
            sample = {
                key: self.all_index_value_maps[key][i] for key in self.dataset.tensors
            }
            samples.append(sample)

        if self.transform:
            self.processed_samples = self.map(
                _apply_transform, repeat(self.transform), samples
            )
        else:
            self.processed_samples = samples
        self.processed_range = slice(first_index, last_index)

    def _shared_memory_clean_up(self):
        """Cleans up possibly leaked memory at the end of iteration"""
        for key in self.dataset.tensors:
            chunk_names = self.all_chunk_sets[key]
            _shared_memory_clear(chunk_names)
