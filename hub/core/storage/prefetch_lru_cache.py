import numpy as np
from itertools import repeat
from pathos.pools import ProcessPool  # type: ignore
from typing import Callable, Dict, Optional, Sequence, Tuple, Union, List

from hub.constants import EMERGENCY_STORAGE_PATH, MB
from hub.core.chunk import Chunk
from hub.core.chunk_engine import ChunkEngine
from hub.core.storage import (
    S3Provider,
    LRUCache,
    StorageProvider,
    MemoryProvider,
    SharedMemoryProvider,
    LocalProvider,
)
from hub.util.exceptions import (
    DatasetUnsupportedSharedMemoryCache,
    TensorDoesNotExistError,
)
from hub.util.remove_cache import get_base_storage
from hub.util.prefetch_cache import read_and_store_chunk_groups
from hub.util.iterable_ordered_dict import IterableOrderedDict


class PrefetchLRUCache(LRUCache):
    """Creates a cache that fetches multiple chunks parallelly."""

    def __init__(
        self,
        cache_storage: SharedMemoryProvider,
        next_storage: Optional[StorageProvider],
        cache_size: int,
        dataset,
        num_workers: int,
        tensor_keys: Optional[Sequence[str]],
        transform: Callable,
        mode: Optional[str] = None,
    ):
        super().__init__(cache_storage, next_storage, cache_size)
        self.mode = mode
        self.transform = transform
        self.all_indexes = self._extract_indexes_from_dataset(dataset)
        self.tensor_keys = self._get_tensor_keys(tensor_keys, dataset)
        self.workers = num_workers
        self.map = ProcessPool(nodes=num_workers).map

        # shared memory file names have format "al_{x}" where x is last_shm_key_generated, which is incremented by 1 every time
        self.last_shm_key_generated = -1

        # keeps track of the last index suggested from all_indexes, incremented by 1 every time to return sequential indexes
        self.last_index_suggested = -1
        self.length = len(dataset)

        self.storage = get_base_storage(dataset.storage)
        if isinstance(self.storage, MemoryProvider):
            raise DatasetUnsupportedSharedMemoryCache(
                "The underlying storage is MemoryProvider which isn't supported."
            )
        elif isinstance(self.storage, S3Provider):
            self.storage_state_tuple = self.storage.__getstate__()

        # map from tuple (tensor, chunk_name) to shared_memory_key
        self.chunk_shared_mem_map: Dict[tuple, str] = {}

        # map from shared_memory_key to (tensor, chunk_name)
        self.shared_mem_chunk_map: Dict[str, tuple] = {}

        # map from each index to a dictionary having tensors as keys and chunk_names as values
        self.index_chunk_names_map: Dict[int, Dict[str, str]] = {}

        self.all_chunk_engines: Dict[str, ChunkEngine] = self._load_all_chunk_engines()

        # chunks that are needed for the current index, these should not be removed from cache. If cache is too small and next storage doesn't exist, it sends to emergency storage
        self.required_chunks: List[tuple] = []

        self.emergency_storage = (
            LocalProvider(EMERGENCY_STORAGE_PATH) if self.next_storage is None else None
        )

    def __getitem__(self, path):
        if path in self.lru_sizes:
            self.lru_sizes.move_to_end(path)  # refresh position for LRU
            return self.cache_storage[path]
        elif self.next_storage:
            # fetch from next storage, may throw KeyError
            result = self.next_storage[path]
            if len(result) <= self.cache_size:  # insert in cache if it fits
                self._insert_in_cache(path, result)
            return result
        else:
            # fetch from emergency storage, may throw KeyError
            result = self.emergency_storage[path]
            if len(result) <= self.cache_size:  # insert in cache if it fits
                self._insert_in_cache(path, result)
            return result

    def __len__(self):
        # TODO: changes this
        return self.length

    def iterate_samples(self, yield_index: bool = False):
        """Iterates over the contents of the dataset and yields data indexwise. If yield_index is True, the index is also returned with the data."""
        chunk_groups_to_fetch: List[List[Tuple[str, str]]] = []
        pending_indexes: List[int] = []
        for i in range(self.length):
            index = self._suggest_next_index()
            chunk_names_dict = self._get_chunk_names_for_index(index)
            self.index_chunk_names_map[index] = chunk_names_dict
            chunks_not_found = self._process_chunks_names_dict(chunk_names_dict)

            if chunks_not_found:
                pending_indexes.append(index)
                chunk_groups_to_fetch.append(chunks_not_found)
                if len(chunk_groups_to_fetch) == self.workers or i == len(self) - 1:
                    self._fetch_and_store_required_data(chunk_groups_to_fetch)
                    for index in pending_indexes:
                        yield self.output_for_index(index, yield_index)
                    pending_indexes.clear()
                    self.required_chunks.clear()
                    self.emergency_storage.clear()
            else:
                yield self.output_for_index(index, yield_index)

        self.clear_cache()

    def output_for_index(self, index: int, yield_index: bool = False):
        """Returns the final output for the given index after converting to IterableOrderedDict and transforming."""
        data = self._data_for_index(index)
        sample = IterableOrderedDict((key, data[key]) for key in self.tensor_keys)
        transformed_data = self._apply_transform(sample)
        if yield_index:
            return index, transformed_data
        else:
            return transformed_data

    def clear_cache(self):
        """Flushes the content of all the cache layers if not in read mode and and then deletes contents of all the layers of it.
        This doesn't delete data from the actual storage.
        """
        self._flush_if_not_read_only()
        self.cache_used = 0
        self.lru_sizes.clear()
        self.dirty_keys.clear()
        self.cache_storage.clear()

        if self.next_storage is not None and hasattr(self.next_storage, "clear_cache"):
            self.next_storage.clear_cache()

    def _get_all_chunks_start_end_index(self) -> Dict[str, Dict[str, Tuple[int, int]]]:
        """Gets the start and end indexes present in each chunk across all tensors."""
        all_tensors_mapping = {}
        for tensor, chunk_engine in self.all_chunk_engines.items():
            array = chunk_engine.chunk_id_encoder.array
            current_tensor_mapping = {}
            start_index = 0
            for item in array:
                chunk_id = item[0]
                chunk_name = chunk_engine.chunk_id_encoder.name_from_id(chunk_id)
                end_index = item[1]
                current_tensor_mapping[chunk_name] = (start_index, end_index)
                start_index = end_index + 1
            all_tensors_mapping[tensor] = current_tensor_mapping
        return all_tensors_mapping

    def _suggest_next_index(self) -> int:
        """Suggests the next index to return data from, in prefetch cache this always goes sequentially over all_indexes"""
        self.last_index_suggested += 1
        return self.all_indexes[self.last_index_suggested]

    def _get_tensor_keys(
        self, tensor_keys: Optional[Sequence[str]], dataset
    ) -> List[str]:
        """Sanitizes tensor_keys if not None, else returns all the keys present in the dataset."""
        if tensor_keys is None:
            tensor_keys = list(dataset.tensors)
        else:
            for t in tensor_keys:
                if t not in dataset.tensors:
                    raise TensorDoesNotExistError(t)
            tensor_keys = list(tensor_keys)
        return tensor_keys

    def _extract_indexes_from_dataset(self, dataset):
        """Returns a list of all the indexes in the dataset."""
        tensor_lengths = [len(tensor) for tensor in dataset.tensors.values()]
        length = min(tensor_lengths, default=0)
        return list(dataset.index.values[0].indices(length))

    def _update_cache_insertion(self, chunk_sizes_dict) -> None:
        """Updates the cache after chunks are inserted into it across processes."""
        for key in chunk_sizes_dict:
            tensor, chunk_name = self.shared_mem_chunk_map[key]
            self.required_chunks.append((tensor, chunk_name))
            self.update_used_cache_for_path(key, chunk_sizes_dict[key])
            self.dirty_keys.add(key)
            if hasattr(self, "_update_count_dicts_insertion"):
                self._update_count_dicts_insertion(tensor, chunk_name)

    def _get_chunk_names_for_index(self, index) -> Dict[str, List[str]]:
        """Returns names of all chunks across tensors that have this index"""
        if index in self.index_chunk_names_map:
            return self.index_chunk_names_map[index]
        chunk_names: Dict[int, List[str]] = {}
        for key in self.tensor_keys:
            chunk_engine = self.all_chunk_engines[key]
            names = chunk_engine.get_chunk_names_for_index(index)
            chunk_names[key] = names
        return chunk_names

    def _load_all_chunk_engines(self):
        """Loads chunk engine for all tensors."""
        # creating a cache around base storage to pass to ChunkEngine
        cache = LRUCache(MemoryProvider(), self.storage, 32 * MB)
        return {key: ChunkEngine(key, cache) for key in self.tensor_keys}

    def _numpy_from_chunks(self, index: int, key: str, chunks: List[Chunk]):
        """Takes a list of chunks and returns a numpy array from it"""
        # TODO: separate out casting
        chunk_engine = self.all_chunk_engines[key]

        # TODO: update this once we support images spanning across multiple chunks
        chunk = chunks[0]
        if self.mode != "pytorch":
            return chunk_engine.read_sample_from_chunk(index, chunk, cast=True)
        else:
            import torch

            value = chunk_engine.read_sample_from_chunk(index, chunk, cast=False)
            # typecast if incompatible with pytorch
            dtype = chunk_engine.tensor_meta.dtype
            compatible_dtypes = {
                "uint16": "int32",
                "uint32": "int64",
                "uint64": "int64",
            }
            dtype = compatible_dtypes.get(dtype, dtype)
            try:
                torch_dtype = getattr(torch, np.dtype(dtype).name)  # type: ignore
            except AttributeError:
                raise TypeError(f"Dtype {dtype} is not supported by pytorch.")
            return torch.as_tensor(value.astype(dtype), dtype=torch_dtype)  # type: ignore

    def _chunks_from_chunk_names(self, tensor: str, chunk_names: List[str]):
        """Takes a list of chunk names and returns a list with corresponding chunk objects"""
        shm_names = [self.chunk_shared_mem_map[(tensor, name)] for name in chunk_names]
        chunk_data = [self[shm_name] for shm_name in shm_names]
        return [Chunk.frombuffer(data) for data in chunk_data]

    def _data_for_index(self, index):
        """Returns all the data for a given index"""
        data: Dict[str, np.ndarray] = {}
        chunk_names_dict = self._get_chunk_names_for_index(index)
        for tensor, chunk_names in chunk_names_dict.items():
            chunks = self._chunks_from_chunk_names(tensor, chunk_names)
            arr = self._numpy_from_chunks(index, tensor, chunks)
            data[tensor] = arr
        return data

    def _generate_shared_memory_names(self, chunk_groups: List[List[Tuple[str, str]]]):
        """Generates shared memory names for all chunks in chunk_groups as chunks names often get too large for some OS"""
        for chunk_group in chunk_groups:
            for chunk in chunk_group:
                if chunk not in self.chunk_shared_mem_map:
                    self.last_shm_key_generated += 1
                    shared_memory_name = f"al_{self.last_shm_key_generated}"
                    self.chunk_shared_mem_map[chunk] = shared_memory_name
                    self.shared_mem_chunk_map[shared_memory_name] = chunk

    def _refresh_chunk_in_cache(self, tensor: str, chunk_name: str):
        """Refreshes the postion of the chunk in the cache. Will fail if chunk doesn't exist already."""
        path = self.chunk_shared_mem_map[(tensor, chunk_name)]
        if path in self.lru_sizes:
            self.lru_sizes.move_to_end(path)
        elif self.next_storage:
            result = self.next_storage[path]  # fetch from storage, may throw KeyError
            if len(result) <= self.cache_size:  # insert in cache if it fits
                self._insert_in_cache(path, result)
        else:
            # fetch from emergency storage, may throw KeyError
            result = self.emergency_storage[path]
            if len(result) <= self.cache_size:  # insert in cache if it fits
                self._insert_in_cache(path, result)

    def _pop_from_cache(self) -> None:
        """Pops the least recently used item from the cache."""
        key, itemsize = self.lru_sizes.popitem(last=False)
        if key in self.dirty_keys and self.next_storage is not None:
            self._forward(key, remove_from_dirty=True)
        else:
            if self.shared_mem_chunk_map[key] in self.required_chunks:
                self.emergency_storage[key] = self.cache_storage[key]
            tensor, chunk_name = self.shared_mem_chunk_map[key]
            if hasattr(self, "_update_count_dicts_pop"):
                self._update_count_dicts_pop(tensor, chunk_name)

        del self.cache_storage[key]
        self.cache_used -= itemsize

    def _process_chunks_names_dict(
        self, chunk_names_dict: Dict[str, List[str]]
    ) -> List[str]:
        """Processes the chunk names dictionary and returns names of chunks that need to be fetched"""
        chunks_not_found = []
        for tensor, chunk_names in chunk_names_dict.items():
            for chunk_name in chunk_names:
                chunk = (tensor, chunk_name)
                shm_name = self.chunk_shared_mem_map.get(chunk)
                if shm_name is None or shm_name not in self._list_keys():
                    chunks_not_found.append((tensor, chunk_name))
                else:
                    self.required_chunks.append(chunk)
                    self._refresh_chunk_in_cache(tensor, chunk_name)
        return chunks_not_found

    def _fetch_and_store_required_data(self, chunk_groups: List[List[Tuple[str, str]]]):
        """Generates shared memory names for required data, fetches, stores it and updates cache storage."""
        self._generate_shared_memory_names(chunk_groups)
        chunk_sizes_dict = self._fetch_chunks(chunk_groups)
        self._update_cache_insertion(chunk_sizes_dict)
        if isinstance(self.cache_storage, SharedMemoryProvider):
            self.cache_storage.update_files(list(chunk_sizes_dict.keys()))
        chunk_groups.clear()

    def _fetch_chunks(
        self, chunk_groups: List[List[Tuple[str, str]]]
    ) -> Dict[str, int]:
        """Takes a list of list of key, chunk_name tuples and fetches chunks for each sublist parallely."""
        # fetch chunks from storage in a multiprocessed manner and decompresses them
        storage: Union[S3Provider, Dict] = self.storage
        shared_memory_groups: List[List[str]] = []
        for chunk_group in chunk_groups:
            names = [self.chunk_shared_mem_map[chunk] for chunk in chunk_group]
            shared_memory_groups.append(names)
        # s3 provider is not sent as storage provider but instead sent as a tuple containing it's state
        if isinstance(storage, S3Provider):
            storage = self.storage_state_tuple

        all_chunk_sizes: List[Dict[str, int]] = self.map(
            read_and_store_chunk_groups,
            chunk_groups,
            shared_memory_groups,
            repeat(storage),
        )
        combined_chunk_sizes_dict: Dict[str, int] = {}
        for chunk_sizes in all_chunk_sizes:
            combined_chunk_sizes_dict.update(chunk_sizes)
        return combined_chunk_sizes_dict

    def _apply_transform(self, sample: Union[Dict, Tuple]):
        """Used to apply transform to a single sample"""
        return self.transform(sample) if self.transform else sample
