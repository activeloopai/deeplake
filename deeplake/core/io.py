from abc import abstractmethod, ABC
from random import shuffle
from typing import Dict, Iterator, List, Optional, Sequence, Union
from itertools import cycle
from copy import copy
from warnings import warn
from numpy import nditer, argmin
from numpy import array as nparray
from math import floor
import numpy as np

from deeplake.constants import MB
from deeplake.core.chunk.base_chunk import BaseChunk
from deeplake.core.chunk_engine import ChunkEngine
from deeplake.core.linked_chunk_engine import LinkedChunkEngine
from deeplake.core.meta.encode.base_encoder import LAST_SEEN_INDEX_COLUMN
from deeplake.core.meta.encode.chunk_id import CHUNK_ID_COLUMN, ChunkIdEncoder
from deeplake.core.meta.tensor_meta import TensorMeta
from deeplake.core.storage import (
    LRUCache,
    MemoryProvider,
    StorageProvider,
    LocalProvider,
)
from deeplake.core.tiling.deserialize import combine_chunks
from deeplake.integrations.pytorch.common import check_tensors, validate_decode_method
from deeplake.util.exceptions import (
    DatasetUnsupportedPytorch,
    SampleDecompressionError,
)
from deeplake.util.keys import get_chunk_key, get_tensor_meta_key
from deeplake.util.remove_cache import get_base_storage
from deeplake.util.storage import get_pytorch_local_storage
from PIL import Image  # type: ignore


ChunkEngineMap = Dict[str, ChunkEngine]
CachesMap = Dict[str, LRUCache]


class IOBlock:
    """
    Represents ordered sequential read of samples from corresponding tensor chunks.
    """

    def __init__(self, chunks: List[List[Optional[str]]], indexes: List[int]) -> None:
        self._chunks: List[List[Optional[str]]] = chunks
        self._ind: List[int] = indexes

    def shuffle(self):
        r"""
        Shuffle sequence in which indices would be read from the IOBlock
        """
        shuffle(self._ind)

    def chunk_names(self, tensor_index: int) -> List[Optional[str]]:
        return self._chunks[tensor_index]

    def indices(self) -> List[int]:
        return self._ind

    def chunks(self) -> List[List[Optional[str]]]:
        return self._chunks

    def split(self, n) -> List["IOBlock"]:
        k, m = divmod(len(self._ind), n)
        return [
            IOBlock(
                self._chunks, self._ind[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)]
            )
            for i in range(n)
        ]

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
    Splits list of IO blocks in a way, so PyTorch loader would return
    samples in sequence, when started with `num_worker` > 1.

    Scheduler relays on a fact, that PyTorch DataLoader synchronize
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


class DistributedScheduler(Scheduler):
    """Scheduler arrange IOBlocks between multiple processes and ensure equal
    distribution for each. Initial `List[IOBlock]` order is preserved.
    """

    def __init__(self, num_worker: int = 0) -> None:
        super().__init__()
        self.next_scheduler: Optional[Scheduler] = (
            MultiThreadedNaiveScheduler(num_worker) if num_worker > 0 else None
        )

    def schedule(self, jobs: List[IOBlock]) -> List[Schedule]:
        import torch.distributed as dist
        import torch

        assert dist.is_available()
        assert dist.is_initialized()

        world_size = dist.get_world_size()
        rank = dist.get_rank()

        gr = dist.new_group([i for i in range(world_size)], backend="gloo")

        blocks_len: torch.Tensor = torch.tensor([len(j.indices()) for j in jobs])
        all_idx: torch.Tensor = torch.zeros(
            (sum([len(j.indices()) for j in jobs]), 2), dtype=torch.int
        )

        if rank == 0:
            all_idx = all_idx[0 : floor(len(all_idx) / world_size) * world_size, :]
            all_idx[:, 0] = torch.repeat_interleave(
                torch.arange(len(jobs)), blocks_len
            )[: len(all_idx)]
            all_idx[:, 1] = torch.tensor(
                [i for j in jobs for i in j.indices()][: len(all_idx)]
            )

        thread_local_idx: torch.Tensor = torch.zeros(
            (int(len(all_idx) / world_size), 2), dtype=torch.int
        )

        dist.scatter(
            thread_local_idx,
            scatter_list=list(all_idx.chunk(world_size)) if rank == 0 else None,
            src=0,
            group=gr,
        )

        # recombine assigned blocks
        blocks_map: Dict[int, List[int]] = dict()
        for idx in thread_local_idx:
            key = int(idx[0])
            val = int(idx[1])

            if key in blocks_map:
                blocks_map[key].append(val)
            else:
                blocks_map[key] = [val]

        blocks = [IOBlock(jobs[k].chunks(), v) for k, v in blocks_map.items()]

        if self.next_scheduler:
            return self.next_scheduler.schedule(blocks)
        else:
            return [Schedule(blocks)]


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
        tensors: Sequence[str],
        use_local_cache: bool = False,
        return_index: bool = True,
        pad_tensors: bool = False,
        decode_method: Optional[Dict[str, str]] = None,
        tobytes: Union[bool, Sequence[str]] = False,
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

        self.tensors = tensors
        self.pad_tensors = pad_tensors
        self.decode_method = decode_method
        self.return_index = return_index

        jpeg_png_compressed_tensors = check_tensors(self.dataset, tensors)
        raw_tensors, compressed_tensors = validate_decode_method(
            self.decode_method, tensors, jpeg_png_compressed_tensors
        )
        self.raw_tensors = set(raw_tensors)
        self.compressed_tensors = set(compressed_tensors)

        self.chunk_engines: ChunkEngineMap = self._map_chunk_engines(self.tensors)

        self.local_caches: Optional[CachesMap] = (
            ({tensor: self._use_cache(self.local_storage) for tensor in self.tensors})
            if self.local_storage is not None
            else None
        )
        group_index_length = len(self.dataset.group_index)
        if group_index_length:
            group_index_length += 1  # add 1 for the forward slash
        self._group_index_length = group_index_length

    def read(self, schedule: Schedule) -> Iterator:
        for block in schedule._blocks:
            yield from self.stream(block)

    def stream(self, block: IOBlock):
        for idx in block.indices():

            sample = dict()
            valid_sample_flag = True

            for keyid, (key, engine) in enumerate(self.chunk_engines.items()):
                rel_key = key[self._group_index_length :]
                decompress = key not in self.raw_tensors
                to_pil = key in self.compressed_tensors
                chunk_class = engine.chunk_class
                try:

                    chunks: List[BaseChunk] = []
                    c_names = block.chunk_names(keyid)

                    version_state = self.dataset.version_state
                    if c_names == [None]:
                        sample[rel_key] = engine.get_empty_sample()
                        continue
                    for c_name in c_names:
                        commit_id = engine.get_chunk_commit(c_name)
                        c_key = get_chunk_key(
                            version_state["tensor_names"][key],
                            c_name,  # type: ignore
                            commit_id,
                        )
                        if self.local_caches is not None:
                            local_cache = self.local_caches[key]

                            if c_key in local_cache:
                                chunk = local_cache.get_deeplake_object(c_key, chunk_class, meta=engine.chunk_args)  # type: ignore
                            else:
                                chunk = engine.get_chunk(c_key)
                                local_cache[c_key] = chunk

                                # send data to actual storage
                                local_cache._forward(c_key)
                        else:
                            chunk = engine.get_chunk(c_key)
                        chunks.append(chunk)
                    if len(chunks) == 1:
                        data = engine.read_sample_from_chunk(
                            idx, chunk, decompress=decompress, to_pil=to_pil
                        )
                    else:
                        if not decompress:
                            raise NotImplementedError(
                                "`tobytes=True` is not supported by tiled samples as it can cause recompression."
                            )
                        data = combine_chunks(chunks, idx, engine.tile_encoder)
                        if to_pil:
                            data = Image.fromarray(data)  # type: ignore

                    if data is not None:
                        sample[rel_key] = data
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
                if self.return_index:
                    sample["index"] = np.array([idx])
                yield sample

    def _get_block_for_single_sample(self, idx):
        chunks = []
        for engine in self.chunk_engines.values():
            enc = engine.chunk_id_encoder
            try:
                cids = enc[idx]
                cnames = list(map(enc.name_from_id, cids))
                chunks.append(cnames)
            except Exception:
                chunks.append([None])
        return IOBlock(chunks, [idx])

    def _is_continuious(self):
        idx_entry = self.dataset.index.values[0]
        if isinstance(idx_entry.value, slice):
            step = idx_entry.value.step
            if step and step < 1:
                return False
            return True
        else:
            prev = -1
            for idx in idx_entry.indices(self._get_dataset_length()):
                if idx < prev:
                    return False
                prev = idx
            return True

    def list_blocks(self) -> List[IOBlock]:
        if self._is_continuious():
            return self.list_blocks_continuous()
        return self.list_blocks_random()

    def list_blocks_random(self) -> List[IOBlock]:
        return list(map(self._get_block_for_single_sample, self._get_dataset_indices()))

    def _intersection(self, index, low, high):
        if isinstance(index, slice):
            start = index.start or 0
            stop = index.stop or self._get_dataset_length()
            step = index.step
            if step is None or step == 1:

                return list(range(max(start, low), min(stop, high)))
            else:
                if start < low:
                    rm = (low - start) % step
                    start = low + bool(rm) * (step - rm)
                return list(range(start, min(stop, high), step))
        elif isinstance(index, (list, tuple)):
            ret = index[np.searchsorted(index, low) : np.searchsorted(index, high)]
            if isinstance(ret, tuple):
                ret = list(ret)
            return ret
        elif isinstance(index, int):
            return [index] if index >= low and index < high else []
        else:
            raise TypeError(index)

    def list_blocks_continuous(self) -> List[IOBlock]:
        blocks: List[IOBlock] = list()

        chunk_id_encodings = [
            engine.chunk_id_encoder.array for engine in self.chunk_engines.values()
        ]

        iterators = [
            nditer([arr[:, LAST_SEEN_INDEX_COLUMN], arr[:, CHUNK_ID_COLUMN]])  # type: ignore
            for arr in chunk_id_encodings
        ]

        last_idx: int = 0

        check_fn = any if self.pad_tensors else all
        while check_fn([not it.finished for it in iterators]):
            next_it = iterators[
                argmin(nparray([it.value[0] for it in iterators if not it.finished]))
            ]
            next_it_value = int(next_it.value[0])

            if next_it_value >= last_idx:
                chunks: List[List[Optional[str]]] = []
                for it in iterators:
                    if it.finished:
                        chunks.append([None])
                        continue
                    cur_ids = []
                    if it.value[0] == next_it_value:
                        while not it.finished and it.value[0] == next_it_value:
                            cur_ids.append(it.value[1])
                            it.iternext()
                    else:
                        cur_ids.append(it.value[1])
                    cur_chunks: List[Optional[str]] = [
                        ChunkIdEncoder.name_from_id(cid)  # type: ignore
                        for cid in cur_ids
                    ]
                    chunks.append(cur_chunks)

                streamable_ids = self._intersection(
                    self.dataset.index.values[0].value, last_idx, next_it_value + 1
                )

                if streamable_ids:
                    new_block = IOBlock(chunks, streamable_ids)
                    blocks.append(new_block)

                last_idx = next_it_value + 1

        return blocks

    def _use_cache(self, storage: Union[StorageProvider, LRUCache]) -> LRUCache:
        cache = LRUCache(MemoryProvider(), copy(storage), 32 * MB)
        cache.read_only = storage.read_only
        return cache

    def _map_chunk_engines(self, tensors: Sequence[str]) -> Dict[str, ChunkEngine]:
        return {
            name: self._create_chunk_engine(name, self.dataset.version_state)
            for name in tensors
        }

    def _create_chunk_engine(self, tensor_name, version_state):
        tensor_key = version_state["tensor_names"][tensor_name]
        meta_key = get_tensor_meta_key(tensor_key, version_state["commit_id"])
        cache = self._use_cache(self.storage)
        meta = cache.get_deeplake_object(meta_key, TensorMeta)
        if meta.is_link:
            return LinkedChunkEngine(
                tensor_key,
                cache,
                version_state,
                link_creds=self.dataset.link_creds,
            )
        return ChunkEngine(tensor_key, cache, version_state)

    def _get_dataset_length(self):
        version_state = self.dataset.version_state
        tensor_lengths = [
            len(version_state["full_tensors"][version_state["tensor_names"][tensor]])
            for tensor in self.tensors
        ]
        length_fn = max if self.pad_tensors else min
        return length_fn(tensor_lengths, default=0)

    def _get_dataset_indices(self):
        return self.dataset.index.values[0].indices(self._get_dataset_length())
