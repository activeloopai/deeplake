import hub
import warnings
import numpy as np
from typing import Any, Dict, List, Optional, Sequence, Union
from hub.util.casting import intelligent_cast
from hub.constants import DEFAULT_MAX_CHUNK_SIZE, FIRST_COMMIT_ID
from hub.core.chunk.base_chunk import BaseChunk, InputSample
from hub.core.chunk.chunk_compressed_chunk import ChunkCompressedChunk
from hub.core.chunk.sample_compressed_chunk import SampleCompressedChunk
from hub.core.chunk.uncompressed_chunk import UncompressedChunk
from hub.core.fast_forwarding import ffw_chunk_id_encoder
from hub.core.index.index import Index
from hub.core.meta.encode.chunk_id import ChunkIdEncoder
from hub.core.meta.tensor_meta import TensorMeta
from hub.core.storage.lru_cache import LRUCache
from hub.core.version_control.commit_chunk_set import CommitChunkSet
from hub.core.version_control.commit_node import CommitNode
from hub.util.casting import get_dtype
from hub.util.keys import (
    get_chunk_id_encoder_key,
    get_chunk_key,
    get_tensor_commit_chunk_set_key,
    get_tensor_meta_key,
)
from hub.util.version_control import auto_checkout, commit_chunk_set_exists
from hub.util.exceptions import CorruptedMetaError, DynamicTensorNumpyError

# used for warning the user if updating a tensor caused suboptimal chunks
CHUNK_UPDATE_WARN_PORTION = 0.2


class ChunkEngine:
    def __init__(
        self,
        key: str,
        cache: LRUCache,
        version_state: Dict[str, Any],
        meta_cache: LRUCache = None,
    ):
        """Handles creating `Chunk`s and filling them with incoming samples.

        Data delegation:
            All samples must live inside a chunk. No chunks may contain partial samples, only 1 chunk per sample.
            A chunk holds the dynamic information for the samples they contain (like shape and byte ranges).
            For more information on the `Chunk` format, check out the `Chunk` class.

        ChunkIdEncoder:
            The `ChunkIdEncoder` bidirectionally maps samples to the chunk IDs they live in. For more information,
            see `ChunkIdEncoder`'s docstring.

        Example:
            Given:
                Sample sizes: [1 * MB, 1 * MB, 14 * MB, 15 * MB, 15 * MB]
                Min chunk size: 16 * MB
                Max chunk size: 32 * MB


            Basic logic:
                >>> chunks = []
                >>> chunks.append(sum([1 * MB, 1 * MB, 14 * MB, 15 * MB]))  # i=(0, 1, 2, 3)
                >>> chunks[-1]
                31 * MB
                >>> chunks.append(sum([15 * MB]))  # i=(4,)
                >>> chunks[-1]
                15 * MB

            Samples 0, 1, 2, and 3 can be stored in 1 chunk. sample 4 resides in it's own chunk.

            If more samples come later: sizes = [15 * MB, 1 * MB]

            Basic logic:
                >>> len(chunks)
                2
                >>> chunks[-1]
                15 * MB
                >>> chunks[-1] += sum([15 * MB, 1 * MB])  # i=(5, 6)
                >>> chunks[-1]
                31 * MB
                >>> sum(chunks)
                62 * MB
                >>> len(chunks)
                2

            Because our max chunk size is 32 * MB, we try to fit as much data into this size as possible.


        Args:
            key (str): Tensor key.
            cache (LRUCache): Cache for which chunks and the metadata are stored.
            version_state (Dict[str, Any]): The version state of the dataset, includes commit_id, commit_node, branch, branch_commit_map and commit_node_map.
            meta_cache (LRUCache): Cache used for storing non chunk data such as tensor meta and chunk id encoder during transforms in memory.

        Raises:
            ValueError: If invalid max chunk size.
        """

        self.key = key
        self.cache = cache
        self._meta_cache = meta_cache
        self.version_state = version_state
        self.compression = None
        self._last_row = 0
        self.chunk_class = BaseChunk

        if self.tensor_meta.sample_compression:
            self.compression = self.tensor_meta.sample_compression
            self.chunk_class = SampleCompressedChunk
        elif self.tensor_meta.chunk_compression:
            self.compression = self.tensor_meta.chunk_compression
            self.chunk_class = ChunkCompressedChunk
        else:
            self.chunk_class = UncompressedChunk

    @property
    def max_chunk_size(self):
        # no chunks may exceed this
        return (
            getattr(self.tensor_meta, "max_chunk_size", None) or DEFAULT_MAX_CHUNK_SIZE
        )

    @property
    def chunk_args(self):
        return [
            self.min_chunk_size,
            self.max_chunk_size,
            self.tensor_meta,
            self.compression,
        ]

    @property
    def min_chunk_size(self):
        # only the last chunk may be less than this
        return self.max_chunk_size // 2

    @property
    def tensor_meta(self):
        tensor_meta_key = get_tensor_meta_key(self.key, self.version_state["commit_id"])
        return self.meta_cache.get_cachable(tensor_meta_key, TensorMeta)

    @property
    def meta_cache(self) -> LRUCache:
        return self._meta_cache or self.cache

    @property
    def chunk_id_encoder(self) -> ChunkIdEncoder:
        """Gets the chunk id encoder from cache, if one is not found it creates a blank encoder.
        For more information on what `ChunkIdEncoder` is used for, see the `__init__` docstring.

        Raises:
            CorruptedMetaError: If chunk id encoding was corrupted.

        Returns:
            ChunkIdEncoder: The chunk ID encoder handles the mapping between sample indices
                and their corresponding chunks.
        """
        commit_id = self.version_state["commit_id"]
        key = get_chunk_id_encoder_key(self.key, commit_id)
        if not self.chunk_id_encoder_exists:
            enc = ChunkIdEncoder()
            self.meta_cache[key] = enc
            return enc

        enc = self.meta_cache.get_cachable(key, ChunkIdEncoder)
        return enc

    @property
    def commit_chunk_set(self) -> Optional[CommitChunkSet]:
        """Gets the commit chunk set from cache, if one is not found it creates a blank one.

        Returns:
            Optional[CommitChunkSet]: The commit chunk set keeps track of all the chunks present in the current commit, returns None for the first commit.
        """
        commit_id = self.version_state["commit_id"]
        if commit_id == FIRST_COMMIT_ID:
            # the first commit doesn't need a commit chunk set
            return None
        key = get_tensor_commit_chunk_set_key(self.key, commit_id)
        if not self.commit_chunk_set_exists:
            cset = CommitChunkSet()
            self.meta_cache[key] = cset
            return cset

        cset = self.meta_cache.get_cachable(key, CommitChunkSet)
        return cset

    @property
    def commit_chunk_set_exists(self) -> bool:
        return commit_chunk_set_exists(self.version_state, self.meta_cache, self.key)

    @property
    def chunk_id_encoder_exists(self) -> bool:
        try:
            commit_id = self.version_state["commit_id"]
            key = get_chunk_id_encoder_key(self.key, commit_id)
            self.meta_cache[key]
            return True
        except KeyError:
            return False

    @property
    def num_chunks(self) -> int:
        if not self.chunk_id_encoder_exists:
            return 0
        return self.chunk_id_encoder.num_chunks

    @property
    def num_samples(self) -> int:
        if not self.chunk_id_encoder_exists:
            return 0
        return self.chunk_id_encoder.num_samples

    @property
    def last_chunk_key(self) -> str:
        last_chunk_name = self.last_chunk_name
        commit_id = self.get_chunk_commit(last_chunk_name)
        return get_chunk_key(self.key, last_chunk_name, commit_id)

    @property
    def last_chunk_name(self) -> str:
        return self.chunk_id_encoder.get_name_for_chunk(-1)

    def last_chunk(self) -> Optional[BaseChunk]:
        if self.num_chunks == 0:
            return None
        chunk_name = self.last_chunk_name
        chunk_commit_id = self.get_chunk_commit(chunk_name)
        chunk_key = get_chunk_key(self.key, chunk_name, chunk_commit_id)
        chunk = self.get_chunk(chunk_key)
        if chunk_commit_id != self.version_state["commit_id"]:
            chunk = self.copy_chunk_to_new_commit(chunk, chunk_name)
        chunk.key = chunk_key  # type: ignore
        return chunk

    def get_chunk(self, chunk_key: str) -> BaseChunk:
        return self.cache.get_cachable(
            chunk_key, self.chunk_class, meta=self.chunk_args
        )

    def copy_chunk_to_new_commit(self, chunk, chunk_name):
        """Copies the chunk to the current commit.

        Returns the copied chunk.
        """
        new_chunk_key = get_chunk_key(
            self.key, chunk_name, self.version_state["commit_id"]
        )
        chunk = chunk.copy(self.chunk_args)
        chunk.key = new_chunk_key
        self.cache[new_chunk_key] = chunk
        if self.commit_chunk_set is not None:
            self.commit_chunk_set.add(chunk_name)
        return chunk

    def get_chunk_commit(self, chunk_name) -> str:
        """Returns the commit id that contains the chunk_name."""
        cur_node: Optional[CommitNode] = self.version_state["commit_node"]
        while cur_node is not None:
            commit_id = cur_node.commit_id
            chunk_set_key = get_tensor_commit_chunk_set_key(self.key, commit_id)
            try:
                # the first commit doesn't contain a chunk set, don't repeatedly try to fetch from storage
                if commit_id == FIRST_COMMIT_ID:
                    chunk_set = set()
                else:
                    chunk_set = self.meta_cache.get_cachable(
                        chunk_set_key, CommitChunkSet
                    ).chunks
            except Exception:
                chunk_set = set()
            if chunk_name in chunk_set:
                return commit_id
            cur_node = cur_node.parent  # type: ignore
        # the first commit doesn't have a commit chunk set, so any chunk that wasn't found belongs to the first commit
        return FIRST_COMMIT_ID

    def _write_initialization(self):
        self.cache.check_readonly()
        # if not the head node, checkout to an auto branch that is newly created
        auto_checkout(self.version_state, self.cache)
        ffw_chunk_id_encoder(self.chunk_id_encoder)

    def _write_finalization(self):
        self._synchronize_cache(chunk_keys=[])
        self.cache.maybe_flush()

    def _convert_to_list(self, samples):
        if self.chunk_class != UncompressedChunk:
            return True
        elif isinstance(samples, np.ndarray):
            return samples[0].nbytes >= self.min_chunk_size
        return True

    def _sanitize_samples(self, samples):
        check_samples_type(samples)
        if self.tensor_meta.dtype is None:
            self.tensor_meta.set_dtype(get_dtype(samples))
        if self._convert_to_list(samples):
            samples = list(samples)
        return samples

    def extend(self, samples):
        self._write_initialization()
        samples = self._sanitize_samples(samples)

        tensor_meta = self.tensor_meta
        if tensor_meta.dtype is None:
            tensor_meta.set_dtype(get_dtype(samples))

        current_chunk = self.last_chunk() or self._create_new_chunk()
        updated_chunks = {current_chunk}

        enc = self.chunk_id_encoder
        samples = samples.copy()
        if self.tensor_meta.sample_compression:
            samples = list(samples)

        while len(samples) > 0:
            num_samples_added = current_chunk.extend_if_has_space(samples)

            if num_samples_added == 0:
                if not current_chunk.data_bytes:
                    msg = f"Sorry, some samples couldn't fit inside a single chunk of size {self.min_chunk_size}"
                    raise NotImplementedError(msg)

                current_chunk = self._create_new_chunk()
                updated_chunks.add(current_chunk)
            else:
                enc.register_samples(num_samples_added)
                samples = samples[num_samples_added:]

        for chunk in updated_chunks:
            self.cache[chunk.key] = chunk  # type: ignore
        self._write_finalization()

    def _synchronize_cache(self, chunk_keys: List[str] = None):
        """Synchronizes cachables with the cache.

        Args:
            chunk_keys (List[str]): List of chunk keys to be synchronized. If None, only the last chunk will be synchronized. Defaults to None.
        """

        # TODO implement tests for cache size compute
        # TODO: optimize this by storing all of these keys in the chunk engine's state (posixpath.joins are pretty slow)

        # synchronize chunks
        if chunk_keys is None:
            chunk_keys = [self.last_chunk_key]
        for chunk_key in chunk_keys:
            chunk = self.get_chunk(chunk_key)
            self.cache.update_used_cache_for_path(chunk_key, chunk.nbytes)  # type: ignore

        commit_id = self.version_state["commit_id"]

        # synchronize tensor meta
        tensor_meta_key = get_tensor_meta_key(self.key, commit_id)
        self.meta_cache[tensor_meta_key] = self.tensor_meta

        # synchronize chunk ID encoder
        chunk_id_key = get_chunk_id_encoder_key(self.key, commit_id)
        self.meta_cache[chunk_id_key] = self.chunk_id_encoder

        # first commit doesn't have commit chunk set
        if commit_id != FIRST_COMMIT_ID:
            # synchronize current chunk set, all older ones are immutable
            commit_chunk_set_key = get_tensor_commit_chunk_set_key(self.key, commit_id)
            self.meta_cache[commit_chunk_set_key] = self.commit_chunk_set  # type: ignore

    def _create_new_chunk(self):
        """Creates and returns a new `Chunk`. Automatically creates an ID for it and puts a reference in the cache."""

        chunk_id = self.chunk_id_encoder.generate_chunk_id()
        chunk = self.chunk_class(*self.chunk_args)
        chunk_name = ChunkIdEncoder.name_from_id(chunk_id)
        chunk_key = get_chunk_key(self.key, chunk_name, self.version_state["commit_id"])
        if self.commit_chunk_set is not None:
            self.commit_chunk_set.add(chunk_name)
        self.cache[chunk_key] = chunk
        chunk.key = chunk_key
        return chunk

    def update(
        self,
        index: Index,
        samples: Union[np.ndarray, Sequence[InputSample], InputSample],
        operator: Optional[str] = None,
    ):
        """Update data at `index` with `samples`."""
        self._write_initialization()
        if operator is not None:
            return self._update_with_operator(index, samples, operator)

        enc = self.chunk_id_encoder
        updated_chunks = set()

        index_length = index.length(self.num_samples)
        samples = _make_sequence(samples, index_length)
        chunks_nbytes_after_updates = []
        global_sample_indices = tuple(index.values[0].indices(self.num_samples))
        for i, sample in enumerate(samples):
            global_sample_index = global_sample_indices[i]  # TODO!
            chunk = self.get_chunk_for_sample(global_sample_index, copy=True)
            local_sample_index = enc.translate_index_relative_to_chunks(
                global_sample_index
            )
            # tensor_meta.update_shape_interval(shape)
            chunk.update_sample(local_sample_index, sample)
            updated_chunks.add(chunk)

            # only care about deltas if it isn't the last chunk
            if chunk.key != self.last_chunk_key:  # type: ignore
                chunks_nbytes_after_updates.append(chunk.nbytes)

        # TODO: [refactor] this is a hacky way, also `self._synchronize_cache` might be redundant. maybe chunks should use callbacks.
        for chunk in updated_chunks:
            self.cache[chunk.key] = chunk  # type: ignore

        self._write_finalization()

        _warn_if_suboptimal_chunks(
            chunks_nbytes_after_updates, self.min_chunk_size, self.max_chunk_size
        )

    def _update_with_operator(
        self,
        index: Index,
        samples: Union[np.ndarray, Sequence[InputSample], InputSample],
        operator: str,
    ):
        """Update data at `index` with the output of elem-wise operatorion with samples"""
        try:
            if isinstance(samples, hub.core.tensor.Tensor):
                samples = samples.numpy()
            arr = self.numpy(index)
        except DynamicTensorNumpyError:
            raise NotImplementedError(
                "Inplace update operations are not available for dynamic tensors yet."
            )

        dt, ht = self.tensor_meta.dtype, self.tensor_meta.htype
        samples = intelligent_cast(samples, dt, ht)
        getattr(arr, operator)(samples)
        self.update(index, arr)

    def read_bytes_for_sample(self, global_sample_index: int) -> bytes:
        if self.tensor_meta.chunk_compression:
            raise Exception(
                "Cannot retreive original bytes for samples in chunk-wise compressed tensors."
            )
        enc = self.chunk_id_encoder
        chunk = self.get_chunk_for_sample(global_sample_index)
        buffer = chunk.memoryview_data
        if not buffer:
            return b""
        local_sample_index = enc.translate_index_relative_to_chunks(global_sample_index)
        sb, eb = chunk.byte_positions_encoder[local_sample_index]
        return buffer[sb:eb].tobytes()

    def read_sample_from_chunk(
        self,
        global_sample_index: int,
        chunk: BaseChunk,
        cast: bool = True,
        copy: bool = False,
    ) -> np.ndarray:
        enc = self.chunk_id_encoder
        local_sample_index = enc.translate_index_relative_to_chunks(global_sample_index)
        return chunk.read_sample(local_sample_index, cast=cast, copy=copy)

    def numpy(
        self, index: Index, aslist: bool = False
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """Reads samples from chunks and returns as a numpy array. If `aslist=True`, returns a sequence of numpy arrays.

        Args:
            index (Index): Represents the samples to read from chunks. See `Index` for more information.
            aslist (bool): If True, the samples will be returned as a list of numpy arrays. If False, returns a single numpy array. Defaults to False.

        Raises:
            DynamicTensorNumpyError: If shapes of the samples being read are not all the same.

        Returns:
            Union[np.ndarray, List[np.ndarray]]: Either a list of numpy arrays or a single numpy array (depending on the `aslist` argument).
        """
        length = self.num_samples
        last_shape = None
        samples = []

        for global_sample_index in index.values[0].indices(length):
            chunk = self.get_chunk_for_sample(global_sample_index)
            sample = self.read_sample_from_chunk(global_sample_index, chunk)
            shape = sample.shape

            if not aslist and last_shape is not None:
                if shape != last_shape:
                    raise DynamicTensorNumpyError(self.key, index, "shape")

            samples.append(sample)
            last_shape = shape

        return _format_read_samples(samples, index, aslist)

    def _is_index_in_last_row(self, arr, index) -> bool:
        """Checks if `index` is in the self._last_row of of chunk_id_encoder."""
        row = self._last_row
        return arr[row][1] >= index and (row == 0 or arr[row - 1][1] < index)

    def _get_chunk_id_for_index(self, global_sample_index: int):
        """Takes a look at self._last_row and tries to find chunk id without binary search by looking at the current and next row.
        Resorts to binary search if the current and next row don't have global_sample_index in them.
        """
        enc = self.chunk_id_encoder
        found = False
        arr = enc.array
        if self._is_index_in_last_row(arr, global_sample_index):
            chunk_id = arr[self._last_row][0]
            found = True
        elif self._last_row < len(arr) - 1:
            self._last_row += 1
            if self._is_index_in_last_row(arr, global_sample_index):
                chunk_id = arr[self._last_row][0]
                found = True

        if not found:
            chunk_id, self._last_row = enc.__getitem__(global_sample_index, True)
        return chunk_id

    def get_chunk_for_sample(
        self, global_sample_index: int, copy: bool = False
    ) -> BaseChunk:
        """Retrives the `Chunk` object corresponding to `global_sample_index`.
        Args:
            global_sample_index (int): Index relative to the entire tensor representing the sample.
            enc (ChunkIdEncoder): Chunk ID encoder. This is an argument because right now it is
                sub-optimal to use `self.chunk_id_encoder` due to posixpath joins.
            copy (bool): If True and the chunk exists in a different commit to the current commit, it will be copied. Defaults to False.
        Returns:
            BaseChunk: BaseChunk object that contains `global_sample_index`.
        """
        chunk_id = self._get_chunk_id_for_index(global_sample_index)

        chunk_name = ChunkIdEncoder.name_from_id(chunk_id)
        chunk_commit_id = self.get_chunk_commit(chunk_name)
        current_commit_id = self.version_state["commit_id"]
        chunk_key = get_chunk_key(self.key, chunk_name, chunk_commit_id)
        chunk = self.cache.get_cachable(
            chunk_key, self.chunk_class, meta=self.chunk_args
        )
        chunk.key = chunk_key
        if chunk_commit_id != current_commit_id and copy:
            chunk = self.copy_chunk_to_new_commit(chunk, chunk_name)
        return chunk

    def validate_num_samples_is_synchronized(self):
        """Check if tensor meta length and chunk ID encoder are representing the same number of samples.
        Helpful for determining if a user has tampered with the tensor meta or the chunk ID encoder, or if
        the tensor was corruptd.

        Raises:
            CorruptedMetaError: tensor_meta and chunk_id_encoder must have the same num samples.
        """

        tensor_meta_length = self.tensor_meta.length

        # compare chunk ID encoder and tensor meta
        chunk_id_num_samples = (
            self.chunk_id_encoder.num_samples if self.chunk_id_encoder_exists else 0
        )
        if tensor_meta_length != chunk_id_num_samples:
            commit_id = self.version_state["commit_id"]
            tkey = get_tensor_meta_key(self.key, commit_id)
            ikey = get_chunk_id_encoder_key(self.key, commit_id)
            raise CorruptedMetaError(
                f"'{tkey}' and '{ikey}' have a record of different numbers of samples. Got {tensor_meta_length} and {chunk_id_num_samples} respectively."
            )


def _format_read_samples(
    samples: List[np.ndarray], index: Index, aslist: bool
) -> Union[np.ndarray, List[np.ndarray]]:
    """Prepares samples being read from the chunk engine in the format the user expects."""

    samples = index.apply(samples)  # type: ignore

    if aslist and all(map(np.isscalar, samples)):
        samples = [arr.item() for arr in samples]

    samples = index.apply_squeeze(samples)  # type: ignore

    if aslist:
        return samples
    else:
        return np.array(samples)


def check_samples_type(samples):
    if not isinstance(samples, (List, np.ndarray)):
        raise TypeError(f"Cannot extend with samples of type {type(samples)}")


def _make_sequence(
    samples: Union[np.ndarray, Sequence[InputSample], InputSample], index_length: int
) -> Sequence[InputSample]:
    """Make `samples` a sequence of `InputSample`s.

    Args:
        samples (Union[np.ndarray, Sequence[InputSample]]): Incoming samples to be made into a sequence.
        index_length (int): Number of expected samples in the sequence.

    Raises:
        ValueError: If `index_length` is incompatible with the true length of `samples`.

    Returns:
        Sequence[InputSample]: Sequence of `InputSample`s with the same length as `index_length`.
    """

    if index_length == 1:
        if hasattr(samples, "__len__"):
            if len(samples) != 1:  # type: ignore
                samples = [samples]
        elif hasattr(samples, "shape"):
            if len(samples.shape) > 0 and samples.shape[0] != 1:  # type: ignore
                samples = [samples]
        else:
            samples = [samples]

    if hasattr(samples, "__len__"):
        if index_length != len(samples):  # type: ignore
            raise ValueError(
                f"Index length ({index_length}) and length of samples ({len(samples)}) must be equal for updating a tensor."  # type: ignore
            )
    else:
        samples = [samples]

    return samples  # type: ignore


def _warn_if_suboptimal_chunks(
    chunks_nbytes_after_updates: List[int], min_chunk_size: int, max_chunk_size: int
):
    upper_warn_threshold = max_chunk_size * (1 + CHUNK_UPDATE_WARN_PORTION)
    lower_warn_threshold = min_chunk_size * (1 - CHUNK_UPDATE_WARN_PORTION)

    for nbytes in chunks_nbytes_after_updates:
        if nbytes > upper_warn_threshold or nbytes < lower_warn_threshold:
            warnings.warn(
                "After update, some chunks were suboptimal. Be careful when doing lots of updates that modify the sizes of samples by a large amount, these can heavily impact read performance!"
            )
            break
