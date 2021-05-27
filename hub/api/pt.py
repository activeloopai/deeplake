from collections import defaultdict
import os
import pickle
from hub.util.keys import get_index_map_key
from hub.core.chunk_engine.chunker import join_chunks
from hub.core.storage import S3Provider
from pathos.pools import ProcessPool
import numpy as np
from hub.core.chunk_engine.read import read_tensor_meta
from itertools import repeat
from functools import lru_cache
from multiprocessing import shared_memory, resource_tracker


@lru_cache()
def s3_client():
    return S3Provider("s3://snark-test/abc-large-3/")


def transform_data(args):
    transform, data = args
    return transform(data) if transform else data


def shared_memory_clear(chunk_set):
    for chunk_name in chunk_set:
        try:
            shm = shared_memory.SharedMemory(name=chunk_name)
            shm.close()
            shm.unlink()
        except:
            pass


def _read_chunks(chunk_path):
    remove_shm_from_resource_tracker()
    storage = s3_client()
    out = storage[chunk_path]
    shm = shared_memory.SharedMemory(
        create=True, size=len(out), name=chunk_path.split("/")[-1]
    )
    shm.buf[:] = out
    shm.close()
    return


def remove_shm_from_resource_tracker():
    """Monkey-patch multiprocessing.resource_tracker so SharedMemory won't be tracked

    More details at: https://bugs.python.org/issue38119
    """

    def fix_register(name, rtype):
        if rtype == "shared_memory":
            return
        return resource_tracker._resource_tracker.register(self, name, rtype)

    resource_tracker.register = fix_register

    def fix_unregister(name, rtype):
        if rtype == "shared_memory":
            return
        return resource_tracker._resource_tracker.unregister(self, name, rtype)

    resource_tracker.unregister = fix_unregister

    if "shared_memory" in resource_tracker._CLEANUP_FUNCS:
        del resource_tracker._CLEANUP_FUNCS["shared_memory"]


# TODO fix warnings at the end of iteration
def _to_pytorch(dataset, transform=None, workers=1):
    try:
        import torch
    except ModuleNotFoundError:
        raise Exception  # TODO proper exception
        # raise ModuleNotInstalledException("torch")
    global torch
    return TorchDataset(dataset, transform, workers)


class TorchDataset:
    def __init__(self, ds, transform=None, workers=1):
        self.ds = ds  # TODO disable the memory cache
        self.transform = transform
        self.workers = workers
        self.storage = self.ds.provider
        self.map = ProcessPool(nodes=workers).map
        self._load_index_maps()
        self._load_meta()
        self.all_index_value_maps = defaultdict(dict)
        self.last_index_map = {}
        self.first_sample_processed = -1
        self.last_sample_processed = -1
        self.all_chunk_sets = {}
        self.shms = []

    def _load_index_maps(self):
        # TODO there should be an easier way in API to do this
        self.all_index_maps = {}
        for key in self.ds.tensors:
            index_map = pickle.loads(self.storage[get_index_map_key(key)])
            self.all_index_maps[key] = index_map

    def _load_meta(self):
        # TODO there should be an easier way in API to do this
        self.all_meta = {}
        for key in self.ds.tensors:
            meta = read_tensor_meta(key, self.storage)
            if meta["dtype"] == "uint16":
                meta["dtype"] = "int32"
            elif meta["dtype"] in ["uint32", "uint64"]:
                meta["dtype"] = "int64"
            self.all_meta[key] = meta

    def __len__(self):
        return len(self.ds)

    def _get_data_from_chunks(self, start_ind, key, chunk_set):
        dtype = self.all_meta[key]["dtype"]
        index_value_map = {}
        chunk_map = {}
        for chunk_name in chunk_set:
            self.shms.append(shared_memory.SharedMemory(name=chunk_name))
            chunk_map[chunk_name] = self.shms[-1].buf[:]
        for index in range(start_ind, len(self.ds)):
            chunks = []
            index_entry = self.all_index_maps[key][index]
            for chunk_name in index_entry["chunk_names"]:
                if chunk_name not in chunk_map:
                    self.all_index_value_maps[key] = index_value_map
                    self.last_index_map[key] = index - 1
                    return
                chunks.append(chunk_map[chunk_name])

            # TODO replace with function that takes in 'chunks', index_entry and dtype and return np.frombuffer value
            # TODO once we have dynamic shaps probably need to read shape from index_entry

            combined_bytes = join_chunks(
                chunks,
                index_entry["start_byte"],
                index_entry["end_byte"],
            )
            index_value_map[index] = np.frombuffer(combined_bytes, dtype=dtype).reshape(
                index_entry["shape"]
            )
        self.all_index_value_maps[key] = index_value_map
        self.last_index_map[key] = len(self.ds) - 1

    def _process_samples(self):
        first_index = self.last_sample_processed + 1
        last_index = min(self.last_index_map[key] for key in self.ds.tensors)
        samples = []
        for i in range(first_index, last_index + 1):
            sample = {
                key: self.all_index_value_maps[key][i] for key in self.ds.tensors
            }
            samples.append(sample)
        if self.transform:
            self.processed_samples = self.map(
                transform_data, zip(repeat(self.transform), samples)
            )
        else:
            self.processed_samples = samples
        self.first_sample_processed = first_index
        self.last_sample_processed = last_index

    def __getitem__(self, index):
        for key in self.ds.tensors:
            if index != 0 and index == self.last_index_map[key] + 1:
                del self.all_index_value_maps[key]
                chunk_set = self.all_chunk_sets[key]
                shared_memory_clear(chunk_set)

            if index in self.all_index_value_maps[key]:
                continue

            chunk_set = set()
            ind = index
            while len(chunk_set) < self.workers and ind < len(self):
                chunk_names = self.all_index_maps[key][ind]["chunk_names"]
                chunk_set.update(chunk_names)
                ind += 1
                if len(chunk_set) > self.workers:
                    chunk_set -= set(chunk_names)
            
            chunk_paths = [
                os.path.join(key, "chunks", chunk_name) for chunk_name in chunk_set
            ]

            shared_memory_clear(chunk_set)
            self.map(_read_chunks, chunk_paths)
            self._get_data_from_chunks(index, key, chunk_set)
            self.all_chunk_sets[key] = chunk_set

        if index > self.last_sample_processed:
            self._process_samples()
        return self.processed_samples[index - self.first_sample_processed]

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]
