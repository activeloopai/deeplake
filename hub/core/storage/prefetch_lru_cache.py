import warnings
import numpy as np
from itertools import repeat
from pathos.pools import ProcessPool  # type: ignore
from typing import Callable, Dict, Optional, Sequence, Tuple, Union, List, Set

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
    SampleDecompressionError,
    TensorDoesNotExistError,
)
from hub.util.remove_cache import get_base_storage
from hub.util.prefetch_cache import read_and_store_chunk_group
from hub.util.iterable_ordered_dict import IterableOrderedDict


# TODO: fetching from local cache happens on the main thread, this needs to be improved
# TODO: transforms are performed on the main thread, this needs to be improved
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
        transform: Optional[Callable],
        mode: Optional[str] = None,
    ):
        super().__init__(cache_storage, next_storage, cache_size)
        self.mode = mode
        self.transform = transform
        self.tensor_keys = self._get_tensor_keys(tensor_keys, dataset)
        self.all_indexes = self._extract_indexes_from_dataset(dataset, self.tensor_keys)
        self.workers = num_workers
        pool = ProcessPool(nodes=num_workers)
        self.map = pool.map

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
        self.index_chunk_names_map: Dict[int, Dict[str, List[str]]] = {}

        self.all_chunk_engines: Dict[str, ChunkEngine] = self._load_all_chunk_engines(
            dataset.version_state
        )
        self.commit_id = dataset.version_state["commit_id"]

        # chunks that are needed for the current index, these should not be removed from cache. If cache is too small and next storage doesn't exist, it sends to emergency storage
        self.required_chunks: Set[tuple] = set()

        self.emergency_storage = (
            LocalProvider(EMERGENCY_STORAGE_PATH) if self.next_storage is None else None
        )

    def __getitem__(self, path):
        if path in self.lru_sizes:
            self.lru_sizes.move_to_end(path)  # refresh position for LRU
            return self.cache_storage[path]
        elif self.next_storage is not None:
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

    def iterate_samples(self):
        """Iterates over the contents of the dataset and yields data indexwise."""
        # chunk groups to be fetched, each inner list will be handled by a separate worker
        chunk_groups_for_workers: List[List[Tuple[str, str]]] = []

        # a set containing all chunks that are scheduled to be fetched
        scheduled_chunks: Set[Tuple[str, str]] = set()

        # indexes that have been encountered but skipped due to data being unavailable
        pending_indexes: List[int] = []
        for i in range(self.length):
            index = self._suggest_next_index()
            chunk_names = self._get_chunk_names(index)
            self.index_chunk_names_map[index] = chunk_names

            # chunks not found for the current index in cache for current index
            missing_chunks = self._process_chunks_names_dict(chunk_names)

            # chunks not found in cache for the current index and also not scheduled to be fetched by another worker
            needed_chunks = self._get_chunks_needed(missing_chunks, scheduled_chunks)

            if missing_chunks:
                pending_indexes.append(index)
                if needed_chunks:
                    chunk_groups_for_workers.append(needed_chunks)
                if len(chunk_groups_for_workers) == self.workers or i == len(self) - 1:
                    yield from self._yield_pending(
                        chunk_groups_for_workers, pending_indexes
                    )
                    scheduled_chunks.clear()
            else:
                # no missing chunks, so we can just yield the data
                yield self._get_final_output(index)

        if pending_indexes:
            yield from self._yield_pending(chunk_groups_for_workers, pending_indexes)
            scheduled_chunks.clear()

        self.clear_cache()

    def _yield_pending(
        self,
        chunk_groups_for_workers: List[List[Tuple[str, str]]],
        pending_indexes: List[int],
    ):
        """Yields data for the indexes that are pending."""
        self._fetch_and_store_required_data(chunk_groups_for_workers)
        for index in pending_indexes:
            yield self._get_final_output(index)
        pending_indexes.clear()
        self.required_chunks.clear()
        if self.emergency_storage is not None:
            self.emergency_storage.clear()

    def _get_chunks_needed(
        self,
        chunks_not_in_cache: List[Tuple[str, str]],
        scheduled_chunks: Set[Tuple[str, str]],
    ) -> List[Tuple[str, str]]:
        """Returns chunks that are not found in cache and not scheduled to be fetched by another worker."""
        chunks_needed: List[Tuple[str, str]] = []
        for chunk in chunks_not_in_cache:
            if chunk not in scheduled_chunks:
                chunks_needed.append(chunk)
                scheduled_chunks.add(chunk)
        return chunks_needed

    def _get_final_output(self, index: int):
        """Returns the final output for the given index after converting to IterableOrderedDict and transforming."""
        data = self._get_data(index)
        if data is None:
            return None
        sample = IterableOrderedDict((key, data[key]) for key in self.tensor_keys)
        return self._apply_transform(sample)

    def clear_cache(self):
        """Flushes the content of all the cache layers if not in read mode and and then deletes contents of all the layers of it.
        This doesn't delete data from the actual storage.
        """
        self.last_index_suggested = -1
        super().clear_cache()

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

        # Get full path in case of groups
        tensor_keys = [dataset.tensors[k].key for k in tensor_keys]
        return tensor_keys

    def _extract_indexes_from_dataset(self, dataset, tensors):
        """Returns a list of all the indexes in the dataset."""
        tensor_lengths = [
            len(dataset.version_state["full_tensors"][tensor]) for tensor in tensors
        ]
        length = min(tensor_lengths, default=0)
        return list(dataset.index.values[0].indices(length))

    def _update_cache_insertion(self, chunk_sizes_dict: Dict[str, int]):
        """Updates the cache after chunks are inserted into it across processes."""
        for key, chunk_size in chunk_sizes_dict.items():
            tensor, chunk_name = self.shared_mem_chunk_map[key]
            self._free_up_space(chunk_size)
            if self.cache_size - self.cache_used >= chunk_size:
                self.update_used_cache_for_path(key, chunk_size)
                self.dirty_keys.add(key)
                self.required_chunks.add((tensor, chunk_name))
                if hasattr(self, "_update_count_dicts_insertion"):
                    self._update_count_dicts_insertion(tensor, chunk_name)  # type: ignore
            elif self.next_storage is not None:
                self.next_storage[key] = self.cache_storage[key]
                del self.cache_storage[key]
                if hasattr(self, "_update_count_dicts_insertion"):
                    self._update_count_dicts_insertion(tensor, chunk_name)  # type: ignore
            elif self.emergency_storage is not None:
                self.emergency_storage[key] = self.cache_storage[key]
                del self.cache_storage[key]
                # we don't update counts when putting data in emergency storage as it well get cleared

    def _get_chunk_names(self, index) -> Dict[str, List[str]]:
        """Returns names of all chunks across tensors that have this index"""
        if index in self.index_chunk_names_map:
            return self.index_chunk_names_map[index]
        chunk_names: Dict[str, List[str]] = {}
        for key in self.tensor_keys:
            chunk_engine = self.all_chunk_engines[key]
            names = chunk_engine.get_chunk_names_for_index(index)
            chunk_names[key] = names
        return chunk_names

    def _load_all_chunk_engines(self, version_state):
        """Loads chunk engine for all tensors."""
        # creating a cache around base storage to pass to ChunkEngine
        cache = LRUCache(MemoryProvider(), self.storage, 32 * MB)
        return {key: ChunkEngine(key, cache, version_state) for key in self.tensor_keys}

    def _numpy_from_chunks(self, index: int, key: str, chunks: List[Chunk]):
        """Takes a list of chunks and returns a numpy array from it"""
        # TODO: separate out casting
        chunk_engine = self.all_chunk_engines[key]

        # TODO: update this once we support images spanning across multiple chunks
        chunk = chunks[0]
        if self.mode != "pytorch":
            try:
                return chunk_engine.read_sample_from_chunk(
                    index, chunk, cast=True, copy=True
                )
            except SampleDecompressionError:
                warnings.warn(
                    f"Skipping corrupt {chunk_engine.tensor_meta.sample_compression} sample."
                )
                return None
        else:
            # read the chunk and cast it to pytorch tensor with compatible dtype
            import torch

            try:
                value = chunk_engine.read_sample_from_chunk(
                    index, chunk, cast=False, copy=False
                )
            except SampleDecompressionError:
                warnings.warn(
                    f"Skipping corrupt {chunk_engine.tensor_meta.sample_compression} sample."
                )
                return None
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

    def _chunks_from_names(self, tensor: str, chunk_names: List[str]):
        """Takes a list of chunk names and returns a list with corresponding chunk objects"""
        return [self._chunk_from_name(tensor, chunk_name) for chunk_name in chunk_names]

    def _chunk_from_name(self, tensor: str, chunk_name: str):
        """Takes a single chunk name and tensor and returns Chunk"""
        shm_name = self.chunk_shared_mem_map[(tensor, chunk_name)]
        chunk_data = self[shm_name]
        chunk = Chunk.frombuffer(chunk_data, copy=False)
        return chunk

    def _numpy_from_chunk_names(self, tensor, chunk_names, index):
        chunks = self._chunks_from_names(tensor, chunk_names)
        arr = self._numpy_from_chunks(index, tensor, chunks)
        return arr

    def _get_data(self, index: int):
        """Returns all the data for a given index"""
        data: Dict[str, np.ndarray] = {}
        chunk_names_dict = self._get_chunk_names(index)
        for tensor, chunk_names in chunk_names_dict.items():
            arr = self._numpy_from_chunk_names(tensor, chunk_names, index)
            if arr is None:
                return None
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
        elif self.next_storage is not None:
            result = self.next_storage[path]  # fetch from storage, may throw KeyError
            if len(result) <= self.cache_size:  # insert in cache if it fits
                self._insert_in_cache(path, result)
        elif self.emergency_storage is not None:
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
            if (
                self.emergency_storage is not None
                and self.shared_mem_chunk_map[key] in self.required_chunks
            ):
                self.emergency_storage[key] = self.cache_storage[key]
            tensor, chunk_name = self.shared_mem_chunk_map[key]
            if hasattr(self, "_update_count_dicts_pop"):
                self._update_count_dicts_pop(tensor, chunk_name)  # type: ignore

        del self.cache_storage[key]
        self.cache_used -= itemsize

    def _process_chunks_names_dict(
        self, chunk_names_dict: Dict[str, List[str]]
    ) -> List[Tuple[str, str]]:
        """Processes the chunk names dictionary and returns names of chunks that need to be fetched"""
        missing_chunks = []
        all_keys = self._all_keys()
        for tensor, chunk_names in chunk_names_dict.items():
            for chunk_name in chunk_names:
                chunk = (tensor, chunk_name)
                shm_name = self.chunk_shared_mem_map.get(chunk)
                if shm_name is None or shm_name not in all_keys:
                    missing_chunks.append((tensor, chunk_name))
                else:
                    self.required_chunks.add(chunk)
                    self._refresh_chunk_in_cache(tensor, chunk_name)
        return missing_chunks

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

        commit_id = self.commit_id
        all_chunk_sizes: List[Dict[str, int]] = self.map(
            read_and_store_chunk_group,
            chunk_groups,
            shared_memory_groups,
            repeat(storage),
            repeat(commit_id),
        )
        combined_chunk_sizes_dict: Dict[str, int] = {}
        for chunk_sizes in all_chunk_sizes:
            combined_chunk_sizes_dict.update(chunk_sizes)
        return combined_chunk_sizes_dict

    def _apply_transform(self, sample: Union[Dict, Tuple]):
        """Used to apply transform to a single sample"""
        return self.transform(sample) if self.transform else sample
