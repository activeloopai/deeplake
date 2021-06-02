import os
import pickle
import numpy as np
from itertools import repeat
from collections import defaultdict
from hub.util.keys import get_index_map_key
from hub.core.chunk_engine.chunker import join_chunks
from hub.core.chunk_engine.read import read_tensor_meta
from pathos.pools import ProcessPool
from multiprocessing import resource_tracker
from multiprocessing.shared_memory import SharedMemory
from hub.util.exceptions import ModuleNotInstalledException


# TODO make this use shared memory to make on the fly transforms faster. Currently using transform slows us down by 10x
def apply_transform(transform, sample):
    """Used to apply transforms to a single samples"""
    return transform(sample) if transform else sample


def shared_memory_clear(chunk_set):
    """Helper function that checks if an existing SharedMemory exists for any chunk in chunk_set and clears it"""
    for chunk_name in chunk_set:
        try:
            shm = SharedMemory(name=chunk_name)
            shm.close()
            shm.unlink()
        except:
            pass


def read_chunk(chunk, key):
    """Reads a single chunk from the dataset's storage provider and stores it in the SharedMemory"""
    remove_shared_memory_from_resource_tracker()
    chunk_path = os.path.join(key, "chunks", chunk)
    chunk_bytes = _hub_storage_provider[chunk_path]
    chunk = chunk_path.split("/")[-1]
    shm = SharedMemory(create=True, size=len(chunk_bytes), name=chunk)
    shm.buf[:] = chunk_bytes
    shm.close()
    return


def remove_shared_memory_from_resource_tracker():
    """Helper function to fix bug in Python SharedMemory

    Monkey-patch multiprocessing.resource_tracker so SharedMemory won't be tracked
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


def _to_pytorch(dataset, transform=None, workers=1):
    try:
        import torch
    except ModuleNotFoundError:
        raise ModuleNotInstalledException("torch")
    global torch
    global _hub_storage_provider  # global to pass to processes, as not possible to serialize it and send as argument

    # TODO, apparently boto3.client isn't safe for multiprocessing https://github.com/boto/boto3/pull/2848/files
    # could it be working here as we're only reading data?
    _hub_storage_provider = dataset.provider
    return TorchDataset(dataset, transform, workers)


class TorchDataset:
    def __init__(self, dataset, transform=None, workers=1):
        self.dataset = dataset  # TODO disable the memory cache
        self.transform = transform
        self.workers = workers
        self.storage = self.dataset.provider
        self.map = ProcessPool(nodes=workers).map

        # contains index_map for each Tensor
        self.all_index_maps = self.load_index_maps()

        # contains meta for each Tensor
        self.all_meta = self.load_meta()

        # corresponding to each Tensor, stores index-value map. here value is the actual array at the index for the key
        # this essentially acts as our in memory prefetch cache
        self.all_index_value_maps = defaultdict(dict)

        # for each Tensor, stores the last index that was prefetched in the prefetch cache
        self.last_index_map = {}

        # list of all the final samples generated after prefetching and transforming, currently present in memory
        self.processed_samples = None
        self.processed_range = slice(-1, -1)  # start and end range of processed_samples

        # keeps track of names of all chunks across tensors whose data is currently prefetched
        self.all_chunk_sets = {}

        # keeps a pointer to all shared memory objects so that they don't get dereferenced between calls to getitem
        self.all_shared_memory = []

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        for key in self.dataset.tensors:
            # if cache hit
            if index in self.all_index_value_maps[key]:
                continue

            # index is outside prefetch cache, clear it (and fetch more later)
            if index != 0 and index == self.last_index_map[key] + 1:
                del self.all_index_value_maps[key]
                chunk_set = self.all_chunk_sets[key]
                shared_memory_clear(chunk_set)

            chunk_set = self.get_chunk_names(index, key)
            shared_memory_clear(chunk_set)
            self.map(read_chunk, chunk_set, repeat(key))
            self.get_data_from_chunks(index, key, chunk_set)
            self.all_chunk_sets[key] = chunk_set

        # if processed cache miss, process more samples
        if index > self.processed_range.stop:
            self.process_samples()
        return self.processed_samples[index - self.processed_range.start]

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]
        self.clean_up()

    # helper functions
    def load_index_maps(self):
        """Loads index maps for all Tensors into memory"""
        # TODO there should be an easier way in API to do this
        all_index_maps = {}
        for key in self.dataset.tensors:
            index_map = pickle.loads(self.storage[get_index_map_key(key)])
            all_index_maps[key] = index_map
        return all_index_maps

    def load_meta(self):
        """Loads meta for all Tensors into memory"""
        # TODO there should be an easier way in API to do this
        all_meta = {}
        # pytorch doesn't support certain dtypes, which are type casted to another dtype below
        for key in self.dataset.tensors:
            meta = read_tensor_meta(key, self.storage)
            if meta["dtype"] == "uint16":
                meta["dtype"] = "int32"
            elif meta["dtype"] in ["uint32", "uint64"]:
                meta["dtype"] = "int64"
            all_meta[key] = meta
        return all_meta

    def get_chunk_names(self, index, key):
        """Gets chunk names to read in parallel"""
        chunk_set = set()
        index_map = self.all_index_maps[key]
        while len(chunk_set) < self.workers and index < len(self):
            chunk_names = index_map[index]["chunk_names"]
            chunk_set.update(chunk_names)
            index += 1
        return chunk_set

    def np_from_chunk_list(self, index, key, chunks):
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

    def get_data_from_chunks(self, index, key, chunk_set):
        """Extracts data from all the chunks in chunk_set and stores it index wise in a dictionary"""
        self.all_index_value_maps[key] = {}

        chunk_map = {}
        # stores value of chunks previously saved in SharedMemory into memory
        for chunk_name in chunk_set:
            self.all_shared_memory.append(SharedMemory(name=chunk_name))
            chunk_map[chunk_name] = self.all_shared_memory[-1].buf[:]

        # saves np array for each index in memory
        for i in range(index, len(self.dataset)):
            chunks = []
            index_entry = self.all_index_maps[key][i]
            for chunk_name in index_entry["chunk_names"]:
                if chunk_name not in chunk_map:
                    self.last_index_map[key] = i - 1
                    return
                chunks.append(chunk_map[chunk_name])
            self.all_index_value_maps[key][i] = self.np_from_chunk_list(i, key, chunks)

        self.last_index_map[key] = len(self.dataset) - 1

    def process_samples(self):
        """Processes the prefetched values from across tensors into dictionaries.
        These samples may be further processed if a transform is specified.
        """
        first_index = self.processed_range.stop + 1
        # different number of samples are fetched for each tensor, we take the min and process
        last_index = min(self.last_index_map[key] for key in self.dataset.tensors)
        samples = []
        for i in range(first_index, last_index + 1):
            sample = {
                key: self.all_index_value_maps[key][i] for key in self.dataset.tensors
            }
            samples.append(sample)

        if self.transform:
            self.processed_samples = self.map(
                apply_transform, repeat(self.transform), samples
            )
        else:
            self.processed_samples = samples
        self.processed_range = slice(first_index, last_index)

    def clean_up(self):
        """cleans up possibly leaked memory at the end of iteration"""
        for key in self.dataset.tensors:
            chunk_set = self.all_chunk_sets[key]
            shared_memory_clear(chunk_set)
        self.all_shared_memory.clear()
