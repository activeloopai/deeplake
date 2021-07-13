from hub.util.dataset import try_flushing
from hub.constants import MB
from hub.util.keys import get_chunk_key
from hub.core.storage.lru_cache import LRUCache
from hub.core.chunk import Chunk
from hub.core.chunk_engine import ChunkEngine
from hub.core.storage import StorageProvider, S3Provider, MemoryProvider
from hub.core.meta.tensor_meta import TensorMeta
from hub.util.remove_cache import get_base_storage
from itertools import repeat
from collections import defaultdict
from typing import Any, Callable, List, Optional, Set, Dict, Union, Tuple, Sequence
from hub.util.exceptions import (
    DatasetUnsupportedPytorch,
    ModuleNotInstalledException,
    TensorDoesNotExistError,
)
from hub.util.iterable_ordered_dict import IterableOrderedDict
from hub.util.shared_memory import (
    remove_shared_memory_from_resource_tracker,
    clear_shared_memory,
)
from pathos.pools import ProcessPool  # type: ignore
from .common import convert_fn as default_convert_fn, collate_fn as default_collate_fn

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


def _import_torch():
    global torch
    try:
        import torch
    except ModuleNotFoundError:
        raise ModuleNotInstalledException(
            "'torch' should be installed to convert the Dataset into pytorch format"
        )


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
    chunk_key = get_chunk_key(key, chunk_name)
    chunk_bytes = storage[chunk_key]
    chunk_size = len(chunk_bytes)
    shared_memory = SharedMemory(create=True, size=chunk_size, name=shared_memory_name)

    # needs to be sliced as some OS (like macOS) allocate extra space
    shared_memory.buf[:chunk_size] = chunk_bytes
    shared_memory.close()
    return chunk_size


def dataset_to_pytorch(
    dataset,
    transform: Optional[Callable] = None,
    tensors: Optional[Sequence[str]] = None,
    num_workers: int = 1,
    batch_size: Optional[int] = 1,
    drop_last: Optional[bool] = False,
    collate_fn: Optional[Callable] = None,
    pin_memory: Optional[bool] = False,
):
    try_flushing(dataset)
    _import_torch()
    # TODO new pytorch approach doesn't support 0 workers currently
    num_workers = max(num_workers, 1)
    pytorch_ds = TorchDataset(dataset, transform, tensors, num_workers)
    if collate_fn is None:
        collate_fn = default_convert_fn if batch_size is None else default_collate_fn
    return torch.utils.data.DataLoader(  # type: ignore
        pytorch_ds,
        batch_size=batch_size,
        drop_last=drop_last,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )


class TorchDataset:
    def __init__(
        self,
        dataset,
        transform: Optional[Callable] = None,
        tensors: Optional[Sequence[str]] = None,
        num_workers: int = 1,
    ):
        self.transform = transform
        self.num_workers: int = num_workers
        self.map = ProcessPool(nodes=num_workers).map
        self.length = len(dataset)
        if tensors is None:
            self.tensor_keys = list(dataset.tensors)
        else:
            for t in tensors:
                if t not in dataset.tensors:
                    raise TensorDoesNotExistError(t)
            self.tensor_keys = list(tensors)
        self.storage = get_base_storage(dataset.storage)
        if isinstance(self.storage, MemoryProvider):
            raise DatasetUnsupportedPytorch(
                "Datasets whose underlying storage is MemoryProvider are not supported for Pytorch iteration."
            )

        elif isinstance(self.storage, S3Provider):
            self.storage_state_tuple = self.storage.__getstate__()

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

        # mapping of each tensor to corresponding chunk_engine
        self.all_chunk_engines: Dict[str, ChunkEngine] = self._load_all_chunk_engines()

        # stores index-value map for each Tensor where value is the actual array at the index
        # acts as in memory prefetch cache
        self.all_index_value_maps: Dict[str, Dict[int, Any]] = defaultdict(dict)

        # tracks last index that was prefetched in the prefetch cache for each Tensor
        self.last_index_meta: Dict[str, int] = {}

        # in memory processed cache containing all samples generated after prefetching and transforming
        self.processed_samples: List = []
        self.processed_range = slice(-1, -1)  # range of processed_samples

        # keeps track of names of all shared_memory that have data in them
        self.all_shared_memory_names: Dict[str, List[str]] = defaultdict(list)

        self.last_chunk_num_generated = -1

    def __len__(self):
        return self.length

    def __getitem__(self, index: int):
        for key in self.tensor_keys:
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

        sample = self._apply_transform(sample)

        return sample

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]

    def _apply_transform(self, sample: Union[Dict, Tuple]):
        """Used to apply transform to a single sample"""
        return self.transform(sample) if self.transform else sample

    # helper functions
    def _load_all_chunk_engines(self):
        """Loads chunk engine for all tensors."""

        # creating a cache around base storage to pass to ChunkEngine
        return {
            key: ChunkEngine(key, LRUCache(MemoryProvider(), self.storage, 16 * MB))
            for key in self.tensor_keys
        }

    def _load_all_meta(self):
        """Loads meta for all Tensors into memory"""
        all_meta = {}
        # pytorch doesn't support certain dtypes, which are type casted to another dtype implicitly
        for key in self.tensor_keys:
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

        chunk_engine = self.all_chunk_engines[key]
        chunk_names = chunk_engine.get_chunk_names(
            index + self.index_offset, len(self) + self.index_offset, self.num_workers
        )

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

    def _generate_shared_memory_names(self, chunk_names: Set[str]):
        """Generates a name for every chunk_name as chunknames are very large and fail on MacOS"""
        ls = []
        for _ in chunk_names:
            self.last_chunk_num_generated += 1
            ls.append(f"al_{self.last_chunk_num_generated}")
        return ls

    def _numpy_from_chunk(self, index: int, key: str, chunk):
        """Takes a list of chunks and returns a numpy array from it"""
        chunk_engine = self.all_chunk_engines[key]
        value = chunk_engine.read_sample_from_chunk(index, chunk)

        # typecast if incompatible with pytorch
        if value.dtype == "uint16":
            value = value.astype("int32")
        elif value.dtype == "uint32" or value.dtype == "uint64":
            value = value.astype("int64")
        return torch.tensor(value)  # type: ignore

    def _get_data_from_chunks(
        self,
        index: int,
        key: str,
        chunk_names: Set[str],
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
            shared_memory = SharedMemory(name=shared_memory_name)
            chunk = Chunk.frombuffer(shared_memory.buf[:chunk_size])
            chunk_map[chunk_name] = chunk

        # saves np array for each index in memory
        for i in range(index, len(self)):
            actual_index = self.index_offset + i
            # TODO change this once it returns list/set of str
            chunk_engine = self.all_chunk_engines[key]
            chunk_id = chunk_engine.chunk_id_encoder[actual_index]
            chunk_name = chunk_engine.chunk_id_encoder.name_from_id(chunk_id)  # type: ignore
            if chunk_name not in chunk_map:
                self.last_index_meta[key] = i - 1
                return
            chunk = chunk_map[chunk_name]
            self.all_index_value_maps[key][i] = self._numpy_from_chunk(
                actual_index, key, chunk
            )

        self.last_index_meta[key] = len(self) - 1

    def _process_samples(self):
        """Processes the prefetched values from across tensors into dictionaries.
        These samples may be further processed if a transform is specified.
        """
        first_index = self.processed_range.stop + 1
        # different no. of samples are fetched for each tensor, take the min and process
        last_index = min(self.last_index_meta[key] for key in self.tensor_keys)
        samples = []
        for i in range(first_index, last_index + 1):
            sample = IterableOrderedDict(
                (key, self.all_index_value_maps[key][i]) for key in self.tensor_keys
            )
            samples.append(sample)
        self.processed_samples = samples
        self.processed_range = slice(first_index, last_index)

    def _all_shared_memory_clean_up(self):
        """Cleans up possibly leaked memory at the end of iteration across Tensors"""
        for key in self.tensor_keys:
            shared_memory_names = self.all_shared_memory_names[key]
            clear_shared_memory(shared_memory_names)
