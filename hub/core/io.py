from abc import abstractmethod
from hashlib import sha1
from random import shuffle
from typing import Dict, List, Optional, Sequence
from copy import copy
from warnings import warn

from hub.constants import MB
from hub.core.chunk import Chunk
from hub.core.chunk_engine import ChunkEngine
from hub.core.storage import LRUCache, MemoryProvider, StorageProvider, LocalProvider
from hub.util.exceptions import (
    DatasetUnsupportedPytorch,
    SampleDecompressionError,
    TensorDoesNotExistError,
)
from hub.util.keys import get_chunk_key
from hub.util.remove_cache import get_base_storage

IndexMap = Dict[int, List[List[str]]]
ChunkEngineMap = Dict[str, ChunkEngine]


class IOBlock:
    r"""
    Represents ordered sequential read of samples from corresponding tensor chunks.
    """

    def __init__(self) -> None:
        self._ind: List[int] = list()

    def add_samples(self, index: int) -> None:
        self._ind.append(index)

    def shuffle(self):
        r"""
        Shuffle sequence in which indices would be read from the IOBlock
        """
        shuffle(self._ind)

    def split(self):
        r"""
        Splits current IOBlock in half and return other part as new IOBlock
        """
        other = IOBlock()
        mid = len(self._ind) // 2
        left = self._ind[:mid]
        right = self._ind[mid:]

        self._ind = left
        other._ind = right

        return other

    def __len__(self) -> int:
        return len(self._ind)


class Schedule:
    def __init__(self, blocks: List[IOBlock]) -> None:
        self._blocks: List[IOBlock] = blocks

    def shuffle(self):
        r"""
        Shuffle IOBlocks in the schedule as well as each IOBlock
        """
        shuffle(self._blocks)

        for block in self._blocks:
            block.shuffle()

    def __iter__(self):
        return iter(self._blocks)

    def __len__(self):
        return sum(map(len, self._blocks))


class Scheduler:
    @abstractmethod
    def schedule(self, jobs: List[IOBlock]) -> List[Schedule]:
        ...


class SingleThreadScheduler(Scheduler):
    def __init__(self) -> None:
        super().__init__()

    def schedule(self, jobs: List[IOBlock]) -> List[Schedule]:
        return [Schedule(jobs)]


class MultiThreadedNativeScheduler(Scheduler):
    def __init__(self, num_workers: int) -> None:
        super().__init__()
        self.num_workers = num_workers

    def schedule(self, jobs: List[IOBlock]) -> List[Schedule]:
        return [Schedule(blocks) for blocks in self.split(jobs, self.num_workers)]

    def split(self, inlist, n):
        k, m = divmod(len(inlist), n)
        return (
            inlist[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n)
        )


class ShufflingSchedulerWrapper(Scheduler):
    def __init__(self, other: Scheduler) -> None:
        super().__init__()
        self.other: Scheduler = other

    def schedule(self, jobs: List[IOBlock]) -> List[Schedule]:
        schedules = self.other.schedule(jobs)
        for schedule in schedules:
            schedule.shuffle()

        return schedules


class SampleStreaming:
    def __init__(
        self,
        dataset,
        scheduler: Scheduler = SingleThreadScheduler(),
        tensors: Optional[Sequence[str]] = None,
        use_local_cache: bool = False,
    ) -> None:
        self.dataset = dataset
        # TODO: determine path
        self.local_storage: Optional[LocalProvider] = (
            LocalProvider() if use_local_cache else None
        )

        # TODO: copy all meta/info to local_storage
        self.storage = get_base_storage(dataset.storage)
        if isinstance(self.storage, MemoryProvider):
            raise DatasetUnsupportedPytorch(
                "The underlying storage is MemoryProvider which isn't supported."
            )

        self.tensors: Sequence[str] = self._map_tensor_keys(
            dataset=dataset, tensor_keys=tensors
        )
        self.chunk_engines: ChunkEngineMap = self._map_chunk_engines(
            self.storage, self.tensors
        )
        self.local_caches: Optional[Dict[str, LRUCache]] = (
            {tensor: self._use_cache(self.local_storage) for tensor in self.tensors}
            if use_local_cache
            else None
        )
        self.index_map: IndexMap = self._map_index_to_chunks()
        self.scheduler: Scheduler = scheduler

    def read(self, schedule: Schedule):
        for job in schedule._blocks:
            yield from self.stream(job._ind)

    def stream(self, indices: Sequence[int]):
        commit_id = self.dataset.version_state["commit_id"]

        for idx in indices:
            sample = dict()
            valid_sample_flag = True

            for keyid, (key, engine) in enumerate(self.chunk_engines.items()):
                try:
                    c_key = get_chunk_key(key, self.index_map[idx][keyid][0], commit_id)

                    if self.local_storage is not None:
                        if c_key in self.local_storage:  # TODO: optimize
                            chunk = self.local_caches[key].get_cachable(c_key, Chunk)
                        else:
                            chunk = engine.get_chunk(c_key)
                            # handle locking, directly write to local_storage, not to cache
                            self.local_storage[c_key] = chunk.tobytes()
                    else:
                        chunk = engine.get_chunk(c_key)

                    data = engine.read_sample_from_chunk(idx, chunk)

                    if data is not None:
                        sample[key] = data
                    else:
                        valid_sample_flag = False
                        break
                except SampleDecompressionError:
                    warn(
                        f"Skipping corrupt {engine.tensor_meta.sample_compression} sample at dataset.{key}[{idx}]"
                    )
                    valid_sample_flag = False
                    break

            if valid_sample_flag:
                yield sample

    def list_blocks(self) -> List[IOBlock]:
        jobs_dict: Dict[bytes, IOBlock] = dict()

        for idx, chunks in self.index_map.items():
            hashed = self._hash_fetch_request(chunks=chunks)

            if jobs_dict.get(hashed) == None:
                jobs_dict[hashed] = IOBlock()

            jobs_dict[hashed].add_samples(idx)

        return list(jobs_dict.values())

    def _use_cache(self, storage: StorageProvider) -> LRUCache:
        return LRUCache(MemoryProvider(), copy(storage), 32 * MB)

    def _map_chunk_engines(self, tensors: List[str]) -> Dict[str, ChunkEngine]:
        return {
            key: self._create_chunk_engine(key, self.dataset.version_state)
            for key in tensors
        }

    def _create_chunk_engine(self, tensor_key, version_state):
        return ChunkEngine(tensor_key, self._use_cache(self.storage), version_state)

    def _map_index_to_chunks(self) -> IndexMap:
        """Build map of all sequence indices and their corresponding chunk names"""
        tensor_lengths = [
            len(self.dataset.version_state["full_tensors"][tensor])
            for tensor in self.tensors
        ]
        length = min(tensor_lengths, default=0)
        dataset_indices = self.dataset.index.values[0].indices(length)

        return {
            index: [
                engine.get_chunk_names_for_index(index)
                for _, engine in self.chunk_engines.items()
            ]
            for index in dataset_indices
        }

    def _map_tensor_keys(
        self, dataset, tensor_keys: Optional[Sequence[str]]
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
        return [dataset.tensors[k].key for k in tensor_keys]

    def _hash_fetch_request(self, chunks: List[List[str]]) -> bytes:
        """Calculates a hash of chunks"""
        sha = sha1()
        for tensor in chunks:
            for chunk in tensor:
                sha.update(chunk.encode())

        return sha.digest()
