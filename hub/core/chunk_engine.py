import hub
import warnings
import json
import numpy as np
from math import ceil
from typing import Any, Dict, Optional, Sequence, Union, Tuple, List, Set

from hub.compression import get_compression_type, BYTE_COMPRESSION, IMAGE_COMPRESSION
from hub.core.version_control.commit_node import CommitNode  # type: ignore
from hub.core.version_control.commit_chunk_set import CommitChunkSet  # type: ignore
from hub.core.fast_forwarding import ffw_chunk_id_encoder
from hub.core.compression import decompress_array, decompress_bytes
from hub.core.sample import Sample, SampleValue  # type: ignore
from hub.core.meta.tensor_meta import TensorMeta
from hub.core.index.index import Index
from hub.core.storage.lru_cache import LRUCache
from hub.core.chunk import Chunk
from hub.core.meta.encode.chunk_id import ChunkIdEncoder
from hub.core.serialize import serialize_input_samples
from hub.core.compression import compress_multiple, decompress_multiple
from hub.constants import DEFAULT_MAX_CHUNK_SIZE, FIRST_COMMIT_ID


from hub.util.keys import (
    get_chunk_key,
    get_chunk_id_encoder_key,
    get_tensor_meta_key,
    get_tensor_commit_chunk_set_key,
)
from hub.util.exceptions import (
    CorruptedMetaError,
    DynamicTensorNumpyError,
)
from hub.util.json import HubJsonDecoder
from hub.util.casting import get_dtype, intelligent_cast
from hub.util.version_control import auto_checkout, commit, commit_chunk_set_exists


def is_uniform_sequence(samples):
    """Determines if a sequence of samples has uniform type and shape, allowing it to be vectorized by `ChunkEngine.extend`."""
    if len(set(map(type, samples))) != 1:
        # Cannot vectorize sequence with inconsistent types
        return False
    elif any(isinstance(s, np.ndarray) for s in samples):
        # Numpy arrays will only be vectorized if they have the same shape
        return len(set(s.shape for s in samples)) == 1
    elif any(isinstance(s, Sample) for s in samples):
        # Sample objects will not be vectorized
        return False
    else:
        # Scalar samples can be vectorized
        return True


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
        self._last_row = 0

        if self.tensor_meta.chunk_compression:
            # Cache samples in the last chunk in uncompressed form.
            self._last_chunk_uncompressed: List[np.ndarray] = (
                self.last_chunk.decompressed_samples(
                    compression=self.tensor_meta.chunk_compression,
                    dtype=self.tensor_meta.dtype,
                )
                if self.last_chunk
                else []
            )

    @property
    def max_chunk_size(self):
        # no chunks may exceed this
        return (
            getattr(self.tensor_meta, "max_chunk_size", None) or DEFAULT_MAX_CHUNK_SIZE
        )

    @property
    def min_chunk_size(self):
        # only the last chunk may be less than this
        return self.max_chunk_size // 2

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
    def last_chunk(self) -> Optional[Chunk]:
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

    def get_chunk(self, chunk_key: str) -> Chunk:
        chunk = self.cache.get_cachable(chunk_key, Chunk)
        chunk.key = chunk_key
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

    @property
    def last_chunk_key(self) -> str:
        last_chunk_name = self.last_chunk_name
        commit_id = self.get_chunk_commit(last_chunk_name)
        return get_chunk_key(self.key, last_chunk_name, commit_id)

    @property
    def last_chunk_name(self) -> str:
        return self.chunk_id_encoder.get_name_for_chunk(-1)

    @property
    def tensor_meta(self):
        tensor_meta_key = get_tensor_meta_key(self.key, self.version_state["commit_id"])
        return self.meta_cache.get_cachable(tensor_meta_key, TensorMeta)

    def _extend_bytes(
        self,
        buffer: memoryview,
        nbytes: List[int],
        shapes: List[Tuple[int]],
    ) -> List[Chunk]:
        """Treat `buffer` as multiple samples and place them into compressed `Chunk`s."""
        if self.tensor_meta.chunk_compression:
            raise NotImplementedError(
                "_extend_bytes not implemented for tensors with chunk wise compression. Use _append_bytes instead."
            )
        chunk = self.last_chunk
        new_chunk = self._create_new_chunk
        if chunk is None:
            chunk = new_chunk()

        updated_chunks = set()

        # If the first incoming sample can't fit in the last chunk, create a new chunk.
        if nbytes[0] > self.min_chunk_size - chunk.num_data_bytes:
            chunk = self._create_new_chunk()

        max_chunk_size = self.max_chunk_size
        min_chunk_size = self.min_chunk_size
        enc = self.chunk_id_encoder

        while nbytes:  # len(nbytes) is initially same as number of incoming samples.
            num_samples_to_current_chunk = 0
            nbytes_to_current_chunk = 0
            for nb in nbytes:  # len(nbytes) = samples remaining to be added to a chunk

                # Size of the current chunk if this sample is added to it
                chunk_future_size = nbytes_to_current_chunk + nb + chunk.num_data_bytes  # type: ignore
                if chunk_future_size > max_chunk_size:
                    break

                num_samples_to_current_chunk += 1
                nbytes_to_current_chunk += nb
                if (
                    chunk_future_size > min_chunk_size
                ):  # Try to keep chunk size close to min_chunk_size
                    break
            chunk.extend_samples(  # type: ignore
                buffer[:nbytes_to_current_chunk],
                max_chunk_size,
                shapes[:num_samples_to_current_chunk],
                nbytes[:num_samples_to_current_chunk],
            )
            updated_chunks.add(chunk)
            enc.register_samples(num_samples_to_current_chunk)

            # Remove bytes from buffer that have been added to current chunk
            buffer = buffer[nbytes_to_current_chunk:]

            # Remove shapes and nbytes for samples that have beed added to current chunk
            del nbytes[:num_samples_to_current_chunk]
            del shapes[:num_samples_to_current_chunk]

            if buffer:
                chunk = new_chunk()
        return updated_chunks  # type: ignore

    def _append_bytes_to_compressed_chunk(self, buffer: memoryview, shape: Tuple[int]):
        """Treat `buffer` as single sample and place them into compressed `Chunk`s."""
        chunk_compression = self.tensor_meta.chunk_compression
        last_chunk_uncompressed = self._last_chunk_uncompressed

        # Append incoming buffer to last chunk and compress:
        last_chunk_uncompressed.append(
            np.frombuffer(buffer, dtype=self.tensor_meta.dtype).reshape(shape)
        )
        compressed_bytes = compress_multiple(last_chunk_uncompressed, chunk_compression)

        # Check if last chunk can hold new compressed buffer.
        if self._can_set_to_last_chunk(len(compressed_bytes)):
            chunk = self.last_chunk
        else:
            # Last chunk full, create new chunk
            chunk = self._create_new_chunk()

            # All samples except the last one are already in the previous chunk, so remove them from cache and compress:
            del last_chunk_uncompressed[:-1]
            compressed_bytes = compress_multiple(
                last_chunk_uncompressed, chunk_compression
            )

        # Set chunk data
        chunk._data = compressed_bytes  # type: ignore

        # Update headers
        if get_compression_type(chunk_compression) == BYTE_COMPRESSION:
            chunk.register_sample_to_headers(incoming_num_bytes=len(buffer), sample_shape=shape)  # type: ignore
        else:
            # Byte positions are not relevant for image compressions, so incoming_num_bytes=None.
            chunk.register_sample_to_headers(incoming_num_bytes=None, sample_shape=shape)  # type: ignore
        return chunk

    def _append_bytes(self, buffer: memoryview, shape: Tuple[int]) -> Chunk:
        """Treat `buffer` as a single sample and place them into `Chunk`s. This function implements the algorithm for
        determining which chunks contain which parts of `buffer`.

        Args:
            buffer (memoryview): Buffer that represents a single sample. Can have a
                length of 0, in which case `shape` should contain at least one 0 (empty sample).
            shape (Tuple[int]): Shape for the sample that `buffer` represents.

        Returns:
            Chunk to which the sample was appended.
        """

        # num samples is always 1 when appending
        num_samples = 1

        if self.tensor_meta.chunk_compression:
            chunk_after_append = self._append_bytes_to_compressed_chunk(buffer, shape)
        else:
            chunk_after_append = self._try_appending_to_last_chunk(buffer, shape)
            if not chunk_after_append:
                chunk_after_append = self._append_to_new_chunk(buffer, shape)

        self.chunk_id_encoder.register_samples(num_samples)
        return chunk_after_append

    def _can_set_to_last_chunk(self, nbytes: int) -> bool:
        """Whether last chunk's data can be set to a buffer of size nbytes."""
        last_chunk = self.last_chunk
        if last_chunk is None:
            return False
        return nbytes <= self.min_chunk_size

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

    def _try_appending_to_last_chunk(self, buffer: memoryview, shape: Tuple[int]):
        """Will store `buffer` inside of the last chunk if it can.
        It can be stored in the last chunk if it exists and has space for `buffer`.

        Args:
            buffer (memoryview): Data to store. This can represent any number of samples.
            shape (Tuple[int]): Shape for the sample that `buffer` represents.

        Returns:
            Last chunk if `buffer` was successfully written to the last chunk, otherwise False.
        """

        last_chunk = self.last_chunk
        if last_chunk is None:
            return False

        incoming_num_bytes = len(buffer)

        if last_chunk.is_under_min_space(self.min_chunk_size):
            last_chunk_size = last_chunk.num_data_bytes
            chunk_ct_content = _min_chunk_ct_for_data_size(
                self.max_chunk_size, incoming_num_bytes
            )

            extra_bytes = min(incoming_num_bytes, self.max_chunk_size - last_chunk_size)
            combined_chunk_ct = _min_chunk_ct_for_data_size(
                self.max_chunk_size, incoming_num_bytes + last_chunk_size
            )

            # combine if count is same
            if combined_chunk_ct == chunk_ct_content:
                last_chunk.append_sample(
                    buffer[:extra_bytes], self.max_chunk_size, shape
                )
                return last_chunk

        return False

    def _append_to_new_chunk(self, buffer: memoryview, shape: Tuple[int]):
        """Will create a new chunk and store `buffer` inside of it. Assumes that `buffer`'s length is < max chunk size.
        This should be called if `buffer` could not be added to the last chunk.

        Args:
            buffer (memoryview): Data to store. This can represent any number of samples.
            shape (Tuple[int]): Shape for the sample that `buffer` represents.

        Returns:
            New chunk that was created
        """

        # check if `last_chunk_extended` to handle empty samples
        new_chunk = self._create_new_chunk()
        new_chunk.append_sample(buffer, self.max_chunk_size, shape)
        return new_chunk

    def _create_new_chunk(self):
        """Creates and returns a new `Chunk`. Automatically creates an ID for it and puts a reference in the cache."""

        chunk_id = self.chunk_id_encoder.generate_chunk_id()
        chunk = Chunk()
        chunk_name = ChunkIdEncoder.name_from_id(chunk_id)
        chunk_key = get_chunk_key(self.key, chunk_name, self.version_state["commit_id"])
        if self.commit_chunk_set is not None:
            self.commit_chunk_set.add(chunk_name)
        self.cache[chunk_key] = chunk
        chunk.key = chunk_key
        return chunk

    def extend(self, samples: Union[np.ndarray, Sequence[SampleValue]]):
        """Formats a batch of `samples` and feeds them into `_append_bytes`."""

        self.cache.check_readonly()
        # if not the head node, checkout to an auto branch that is newly created
        auto_checkout(self.version_state, self.cache)
        ffw_chunk_id_encoder(self.chunk_id_encoder)

        tensor_meta = self.tensor_meta
        if tensor_meta.dtype is None:
            tensor_meta.set_dtype(get_dtype(samples))

        buff, nbytes, shapes = serialize_input_samples(
            samples, tensor_meta, self.min_chunk_size
        )
        for shape in shapes:
            tensor_meta.update_shape_interval(shape)
        tensor_meta.length += len(samples)
        if tensor_meta.chunk_compression:
            updated_chunks = set()
            for nb, shape in zip(nbytes, shapes):
                chunk = self._append_bytes(buff[:nb], shape[:])  # type: ignore
                updated_chunks.add(chunk)
                buff = buff[nb:]
                updated_chunks.add(chunk)
        else:
            updated_chunks = self._extend_bytes(buff, nbytes, shapes[:])  # type: ignore
        for chunk in updated_chunks:
            self.cache[chunk.key] = chunk  # type: ignore
        self._synchronize_cache(chunk_keys=[])
        self.cache.maybe_flush()

    def append(self, sample: SampleValue):
        """Formats a single `sample` (compresseses/decompresses if applicable) and feeds it into `_append_bytes`."""
        self.extend([sample])

    def update(
        self,
        index: Index,
        samples: Union[Sequence[SampleValue], SampleValue],
        operator: Optional[str] = None,
    ):
        """Update data at `index` with `samples`."""

        if operator is not None:
            return self._update_with_operator(index, samples, operator)

        self.cache.check_readonly()
        # if not the head node, checkout to an auto branch that is newly created
        auto_checkout(self.version_state, self.cache)
        ffw_chunk_id_encoder(self.chunk_id_encoder)

        tensor_meta = self.tensor_meta
        enc = self.chunk_id_encoder
        updated_chunks = set()

        index_length = index.length(self.num_samples)
        samples = _make_sequence(samples, index_length)
        serialized_input_samples = serialize_input_samples(
            samples, tensor_meta, self.min_chunk_size
        )

        chunks_nbytes_after_updates = []
        global_sample_indices = tuple(index.values[0].indices(self.num_samples))
        buffer, nbytes, shapes = serialized_input_samples
        for i, (nb, shape) in enumerate(zip(nbytes, shapes)):
            global_sample_index = global_sample_indices[i]  # TODO!
            chunk = self.get_chunk_for_sample(global_sample_index, enc, copy=True)
            local_sample_index = enc.translate_index_relative_to_chunks(
                global_sample_index
            )
            tensor_meta.update_shape_interval(shape)
            chunk.update_sample(
                local_sample_index,
                buffer[:nb],  # type: ignore
                shape,
                chunk_compression=self.tensor_meta.chunk_compression,
                dtype=self.tensor_meta.dtype,
            )
            buffer = buffer[nb:]
            updated_chunks.add(chunk)

            # only care about deltas if it isn't the last chunk
            if chunk.key != self.last_chunk_key:  # type: ignore
                chunks_nbytes_after_updates.append(chunk.nbytes)

        # TODO: [refactor] this is a hacky way, also `self._synchronize_cache` might be redundant. maybe chunks should use callbacks.
        for chunk in updated_chunks:
            self.cache[chunk.key] = chunk  # type: ignore

        self._synchronize_cache(chunk_keys=[])
        self.cache.maybe_flush()

        _warn_if_suboptimal_chunks(
            chunks_nbytes_after_updates, self.min_chunk_size, self.max_chunk_size
        )

    def _update_with_operator(
        self,
        index: Index,
        samples: Union[Sequence[SampleValue], SampleValue],
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
        samples = intelligent_cast(
            samples, self.tensor_meta.dtype, self.tensor_meta.htype
        )
        getattr(arr, operator)(samples)
        self.update(index, arr)

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
            Union[np.ndarray, Sequence[np.ndarray]]: Either a list of numpy arrays or a single numpy array (depending on the `aslist` argument).
        """
        length = self.num_samples
        enc = self.chunk_id_encoder
        last_shape = None
        samples = []

        for global_sample_index in index.values[0].indices(length):
            chunk = self.get_chunk_for_sample(global_sample_index, enc)
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

    def _get_chunk_id_for_index(self, global_sample_index: int, enc: ChunkIdEncoder):
        """Takes a look at self._last_row and tries to find chunk id without binary search by looking at the current and next row.
        Resorts to binary search if the current and next row don't have global_sample_index in them.
        """
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
        self, global_sample_index: int, enc: ChunkIdEncoder, copy: bool = False
    ) -> Chunk:
        """Retrives the `Chunk` object corresponding to `global_sample_index`.
        Args:
            global_sample_index (int): Index relative to the entire tensor representing the sample.
            enc (ChunkIdEncoder): Chunk ID encoder. This is an argument because right now it is
                sub-optimal to use `self.chunk_id_encoder` due to posixpath joins.
            copy (bool): If True and the chunk exists in a different commit to the current commit, it will be copied. Defaults to False.
        Returns:
            Chunk: Chunk object that contains `global_sample_index`.
        """
        chunk_id = self._get_chunk_id_for_index(global_sample_index, enc)

        chunk_name = ChunkIdEncoder.name_from_id(chunk_id)
        chunk_commit_id = self.get_chunk_commit(chunk_name)
        current_commit_id = self.version_state["commit_id"]
        chunk_key = get_chunk_key(self.key, chunk_name, chunk_commit_id)
        chunk = self.cache.get_cachable(chunk_key, Chunk)
        chunk.key = chunk_key
        if chunk_commit_id != current_commit_id and copy:
            chunk = self.copy_chunk_to_new_commit(chunk, chunk_name)
        return chunk

    def read_bytes_for_sample(self, global_sample_index: int) -> bytes:
        if self.tensor_meta.chunk_compression:
            raise Exception(
                "Cannot retreive original bytes for samples in chunk-wise compressed tensors."
            )
        enc = self.chunk_id_encoder
        chunk = self.get_chunk_for_sample(global_sample_index, enc)
        buffer = chunk.memoryview_data
        if not buffer:
            return b""
        local_sample_index = enc.translate_index_relative_to_chunks(global_sample_index)
        sb, eb = chunk.byte_positions_encoder[local_sample_index]
        return buffer[sb:eb].tobytes()

    def read_sample_from_chunk(
        self,
        global_sample_index: int,
        chunk: Chunk,
        cast: bool = True,
        copy: bool = False,
    ) -> np.ndarray:
        """Read a sample from a chunk, converts the global index into a local index. Handles decompressing if applicable."""

        dtype = self.tensor_meta.dtype

        enc = self.chunk_id_encoder

        buffer = chunk.memoryview_data

        local_sample_index = enc.translate_index_relative_to_chunks(global_sample_index)
        shape = chunk.shapes_encoder[local_sample_index]

        if len(buffer) == 0:
            return np.zeros(shape, dtype=dtype)

        chunk_compression = self.tensor_meta.chunk_compression
        sample_compression = self.tensor_meta.sample_compression

        htype = self.tensor_meta.htype

        if htype in ("json", "text", "list"):
            sb, eb = chunk.byte_positions_encoder[local_sample_index]
            if chunk_compression:
                decompressed = chunk.decompressed_data(compression=chunk_compression)
                buffer = decompressed[sb:eb]
            elif sample_compression:
                buffer = decompress_bytes(buffer[sb:eb], compression=sample_compression)
            else:
                buffer = buffer[sb:eb]
            buffer = bytes(buffer)
            if htype == "json":
                arr = np.empty(1, dtype=object)
                arr[0] = json.loads(bytes.decode(buffer), cls=HubJsonDecoder)
                return arr
            elif htype == "list":
                lst = json.loads(bytes.decode(buffer), cls=HubJsonDecoder)
                arr = np.empty(len(lst), dtype=object)
                arr[:] = lst
                return arr
            elif htype == "text":
                arr = np.array(bytes.decode(buffer)).reshape(
                    1,
                )
            return arr

        if chunk_compression:
            if get_compression_type(chunk_compression) == BYTE_COMPRESSION:
                decompressed = chunk.decompressed_data(compression=chunk_compression)
                sb, eb = chunk.byte_positions_encoder[local_sample_index]
                return np.frombuffer(decompressed[sb:eb], dtype=dtype).reshape(shape)
            else:
                return chunk.decompressed_samples()[local_sample_index]

        sb, eb = chunk.byte_positions_encoder[local_sample_index]
        buffer = buffer[sb:eb]
        if sample_compression:
            sample = decompress_array(
                buffer, shape, dtype=dtype, compression=sample_compression
            )
            if cast and sample.dtype != dtype:
                sample = sample.astype(dtype)
        else:
            if copy:
                buffer = bytes(buffer)
            sample = np.frombuffer(buffer, dtype=dtype).reshape(shape)

        return sample

    def get_chunk_names_for_multiple_indexes(
        self, sample_index: int, last_index: int, target_chunk_count: int
    ) -> Set[str]:
        """Fetches a set of chunk names in which data starting from sample_index is contained.
            This is used by Pytorch integration.

        Args:
            sample_index: The index starting from which chunk names need to be fetched.
            last_index: The last index till which chunk names need to be fetched.
            target_chunk_count: The target number of chunk names required. The actual size of the returned set may be:-
                a) Less than target_chunk_count: If there are no more chunks to fetch.
                b) More than target_chunk_count: If the last chunk filling up target_chunk_count is a partial chunk, the remaining chunks are fetched.
                c) Equal to the target_chunk_count: In all other cases.

        Returns:
            Set of chunk names.
        """
        chunk_names: Set[str] = set()
        while len(chunk_names) < target_chunk_count and sample_index < last_index:
            chunk_id = self.chunk_id_encoder[sample_index]
            chunk = self.chunk_id_encoder.name_from_id(chunk_id)
            # todo, change to chunk_names.update once chunks returns sequence instead of single string
            chunk_names.add(chunk)
            sample_index += 1
        return chunk_names

    def get_chunk_names_for_index(self, sample_index):
        # TODO: fix this once we support multiple chunk names per sample
        chunk_id = self.chunk_id_encoder[sample_index]
        chunk = self.chunk_id_encoder.name_from_id(chunk_id)
        return [chunk]

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

    def copy_chunk_to_new_commit(self, chunk, chunk_name):
        """Copies the chunk to the current commit.

        Returns the copied chunk.
        """
        new_chunk_key = get_chunk_key(
            self.key, chunk_name, self.version_state["commit_id"]
        )
        chunk = chunk.copy()
        chunk.key = new_chunk_key
        self.cache[new_chunk_key] = chunk
        if self.commit_chunk_set is not None:
            self.commit_chunk_set.add(chunk_name)
        return chunk


def _format_read_samples(
    samples: Sequence[np.ndarray], index: Index, aslist: bool
) -> Union[np.ndarray, List[np.ndarray]]:
    """Prepares samples being read from the chunk engine in the format the user expects."""

    samples = index.apply(samples)  # type: ignore

    if aslist and all(map(np.isscalar, samples)):
        samples = list(arr.item() for arr in samples)

    samples = index.apply_squeeze(samples)  # type: ignore

    if aslist:
        return samples  # type: ignore
    else:
        return np.array(samples)


def _min_chunk_ct_for_data_size(chunk_max_data_bytes: int, size: int) -> int:
    """Calculates the minimum number of chunks in which data of given size can be fit."""
    return ceil(size / chunk_max_data_bytes)


def _make_sequence(
    samples: Union[Sequence[SampleValue], SampleValue], index_length: int
) -> Sequence[SampleValue]:
    """Make `samples` a sequence of `SampleValue`s.

    Args:
        samples (Union[Sequence[SampleValue], SampleValue]): Incoming samples to be made into a sequence.
        index_length (int): Number of expected samples in the sequence.

    Raises:
        ValueError: If `index_length` is incompatible with the true length of `samples`.

    Returns:
        Sequence[SampleValue]: Sequence of `SampleValue`s with the same length as `index_length`.
    """

    if index_length == 1:
        if hasattr(samples, "__len__"):
            if len(samples) != 1:
                samples = [samples]
        elif hasattr(samples, "shape"):
            if len(samples.shape) > 0 and samples.shape[0] != 1:  # type: ignore
                samples = [samples]
        else:
            samples = [samples]

    if hasattr(samples, "__len__"):
        if index_length != len(samples):
            raise ValueError(
                f"Index length ({index_length}) and length of samples ({len(samples)}) must be equal for updating a tensor."
            )
    else:
        samples = [samples]

    return samples


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
