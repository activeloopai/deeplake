from abc import abstractmethod, ABC
from random import shuffle, randrange
from typing import Dict, Iterator, List, Optional, Sequence, Union
from itertools import cycle
from copy import copy
from warnings import warn
from numpy import nditer, argmin
from numpy import array as nparray


from hub.constants import MB
from hub.core.chunk import Chunk
from hub.core.chunk_engine import ChunkEngine
from hub.core.meta.encode.base_encoder import LAST_SEEN_INDEX_COLUMN
from hub.core.meta.encode.chunk_id import CHUNK_ID_COLUMN, ChunkIdEncoder
from hub.core.storage import LRUCache, MemoryProvider, StorageProvider, LocalProvider
from hub.util.exceptions import (
    DatasetUnsupportedPytorch,
    SampleDecompressionError,
    TensorDoesNotExistError,
)
from hub.util.keys import get_chunk_key
from hub.util.remove_cache import get_base_storage
from hub.util.storage import get_pytorch_local_storage

ChunkEngineMap = Dict[str, ChunkEngine]
CachesMap = Dict[str, LRUCache]


class IOBlock:
    """
    Represents ordered sequential read of samples from corresponding tensor chunks.
    """

    def __init__(self, chunks: List[str], indexes: List[int]) -> None:
        self._chunks: List[str] = chunks
        self._ind: List[int] = indexes

    def shuffle(self):
        r"""
        Shuffle sequence in which indices would be read from the IOBlock
        """
        shuffle(self._ind)

    def chunk_name(self, tensor_index: int) -> str:
        return self._chunks[tensor_index]

    def indices(self) -> List[int]:
        return self._ind

    def chunks(self) -> List[str]:
        return self._chunks

    def __len__(self) -> int:
        return len(self._ind)


class Schedule:
    def __init__(self, blocks: List[IOBlock]) -> None:
        self._blocks: List[IOBlock] = blocks

    def shuffle(self) -> None:
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


class Scheduler(ABC):
    @abstractmethod
    def schedule(self, jobs: List[IOBlock]) -> List[Schedule]:
        ...


class SingleThreadScheduler(Scheduler):
    def schedule(self, jobs: List[IOBlock]) -> List[Schedule]:
        return [Schedule(jobs)]


class SequentialMultithreadScheduler(Scheduler):
    """
    Splits list of IO blocks in a way, so PyTroch loader would return
    samples in sequence, when started with `num_worker` > 1.

    Scheduler relays on a fact, that PyTroch DataLoader synchronize
    read of samples per thread and return in a sequence per worker.

    Example:
        Given sequence of indices `[1, 2, 3, 4, 5, 6]` and 4 workers
        PyTorch have to read samples in an order
            thread0: [1, 5]
            thread1: [2, 6]
            thread2: [3]
            thread3: [4]

        So that initial order would be reconstructed by the DataLoader
    """

    def __init__(self, num_workers: int) -> None:
        super().__init__()
        self.num_workers = num_workers

    def schedule(self, jobs: List[IOBlock]) -> List[Schedule]:
        per_worker: List[List[IOBlock]] = [list() for _ in range(self.num_workers)]
        assigned_worker = iter(cycle(range(self.num_workers)))

        for job in jobs:
            split: List[List[int]] = [list() for _ in range(self.num_workers)]

            for ind in job.indices():
                split[next(assigned_worker)].append(ind)

            for worker_id, idx_list in enumerate(split):
                if len(idx_list) > 0:
                    worker_block = IOBlock(job.chunks(), idx_list)
                    per_worker[worker_id].append(worker_block)

        return [Schedule(worker_jobs) for worker_jobs in per_worker]


class MultiThreadedNaiveScheduler(Scheduler):
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


class Streaming(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def read(self, schedule: Schedule) -> Iterator:
        r"""
        Args:
            schedule(Schedule) schedule of IOBlocks to stream
        Returns:
            generator over specific Schedule
        """
        ...


class SampleStreaming(Streaming):
    def __init__(
        self,
        dataset,
        tensors: Optional[Sequence[str]] = None,
        use_local_cache: bool = False,
    ) -> None:
        super().__init__()

        self.dataset = dataset
        self.local_storage: Optional[LocalProvider] = (
            get_pytorch_local_storage(dataset) if use_local_cache else None
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
        self.chunk_engines: ChunkEngineMap = self._map_chunk_engines(self.tensors)

        self.local_caches: Optional[CachesMap] = (
            ({tensor: self._use_cache(self.local_storage) for tensor in self.tensors})
            if self.local_storage is not None
            else None
        )

    def read(self, schedule: Schedule) -> Iterator:
        for block in schedule._blocks:
            yield from self.stream(block)

    def stream(self, block: IOBlock):
        commit_id = self.dataset.version_state["commit_id"]

        for idx in block.indices():

            sample = dict()
            valid_sample_flag = True

            for keyid, (key, engine) in enumerate(self.chunk_engines.items()):
                try:
                    c_key = get_chunk_key(key, block.chunk_name(keyid), commit_id)
                    chunk: Chunk

                    if self.local_caches is not None:
                        local_cache = self.local_caches[key]

                        if c_key in local_cache:
                            chunk = local_cache.get_cachable(c_key, Chunk)  # type: ignore
                        else:
                            chunk = engine.get_chunk(c_key)
                            local_cache[c_key] = chunk

                            # send data to actual storage
                            local_cache._forward(c_key, True)
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
        blocks: List[IOBlock] = list()

        ds_indicies_set = set(self._get_dataset_indicies())

        chunk_id_encodings = [
            engine.chunk_id_encoder.array for engine in self.chunk_engines.values()
        ]

        iterators = [
            nditer([arr[:, LAST_SEEN_INDEX_COLUMN], arr[:, CHUNK_ID_COLUMN]])  # type: ignore
            for arr in chunk_id_encodings
        ]

        last_idx: int = 0

        while all([not it.finished for it in iterators]):
            next_it = iterators[argmin(nparray([it.value[0] for it in iterators]))]
            next_it_value = int(next_it.value[0])

            if next_it_value >= last_idx:
                chunks = [
                    ChunkIdEncoder.name_from_id(cid)  # type: ignore
                    for cid in [int(it.value[1]) for it in iterators]
                ]

                streamable_ids = list(
                    ds_indicies_set.intersection(range(last_idx, next_it_value + 1))
                )
                streamable_ids.sort()

                if len(streamable_ids) > 0:
                    new_block = IOBlock(chunks, streamable_ids)
                    blocks.append(new_block)

                last_idx = next_it_value + 1

            next(next_it, None)

        return blocks

    def _use_cache(self, storage: Union[StorageProvider, LRUCache]) -> LRUCache:
        return LRUCache(MemoryProvider(), copy(storage), 32 * MB)

    def _map_chunk_engines(self, tensors: List[str]) -> Dict[str, ChunkEngine]:
        return {
            key: self._create_chunk_engine(key, self.dataset.version_state)
            for key in tensors
        }

    def _create_chunk_engine(self, tensor_key, version_state):
        return ChunkEngine(tensor_key, self._use_cache(self.storage), version_state)

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

    def _get_dataset_indicies(self):
        tensor_lengths = [
            len(self.dataset.version_state["full_tensors"][tensor])
            for tensor in self.tensors
        ]
        length = min(tensor_lengths, default=0)

        return self.dataset.index.values[0].indices(length)


class BufferedStreaming(Streaming):
    def __init__(self, streaming: Streaming, size: int) -> None:
        self._streaming = streaming
        self._buffer: List = list()
        self._buffer_size = size

    def read(self, schedule: Schedule):
        buffer = self._buffer
        buffer_size = self._buffer_size

        it = self._streaming.read(schedule)

        # filling buffer with samples
        while len(buffer) < buffer_size:
            data = next(it, None)

            if data is not None:
                buffer.append(data)
            else:
                break

        # stream until all samples exhausted
        while len(buffer) > 0:
            selected = randrange(len(buffer))

            next_val = next(it, None)

            if next_val is not None:
                buffer.append(next_val)

            yield buffer.pop(selected)
