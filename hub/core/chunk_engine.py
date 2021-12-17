import hub
import numpy as np
from typing import Any, Dict, Optional, Sequence, Union, List, Tuple
from hub.core.version_control.commit_diff import CommitDiff
from hub.core.version_control.commit_node import CommitNode  # type: ignore
from hub.core.version_control.commit_chunk_set import CommitChunkSet  # type: ignore
from typing import Any, Dict, List, Optional, Sequence, Union
from hub.core.meta.encode.tile import TileEncoder
from hub.core.tiling.deserialize import combine_chunks
from hub.util.casting import intelligent_cast
from hub.constants import DEFAULT_MAX_CHUNK_SIZE, FIRST_COMMIT_ID, PARTIAL_NUM_SAMPLES
from hub.core.chunk.base_chunk import BaseChunk, InputSample
from hub.core.chunk.chunk_compressed_chunk import ChunkCompressedChunk
from hub.core.chunk.sample_compressed_chunk import SampleCompressedChunk
from hub.core.chunk.uncompressed_chunk import UncompressedChunk
from hub.core.fast_forwarding import ffw_chunk_id_encoder
from hub.core.index.index import Index
from hub.core.meta.encode.chunk_id import ChunkIdEncoder
from hub.core.meta.tensor_meta import TensorMeta
from hub.core.storage.lru_cache import LRUCache
from hub.util.casting import get_dtype
from hub.util.chunk_engine import (
    check_samples_type,
    make_sequence,
    check_suboptimal_chunks,
    format_read_samples,
    check_sample_shape,
)
from hub.util.keys import (
    get_chunk_id_encoder_key,
    get_tensor_commit_diff_key,
    get_tensor_meta_key,
    get_chunk_key,
    get_tensor_commit_chunk_set_key,
    get_tensor_meta_key,
    get_tensor_tile_encoder_key,
)
from hub.util.version_control import auto_checkout, commit_chunk_set_exists
from hub.util.exceptions import CorruptedMetaError, DynamicTensorNumpyError


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

        self.cachables_in_dirty_keys = False

        self.tensor_meta.num_compressed_bytes = self.num_compressed_bytes
        self.tensor_meta.num_uncompressed_bytes = self.num_uncompressed_bytes

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
    def num_compressed_bytes(self):
        nbytes = getattr(self.tensor_meta, "num_compressed_bytes", None)
        if nbytes is None:
            nbytes = self._get_num_compressed_bytes()
        return nbytes

    @property
    def num_uncompressed_bytes(self):
        nbytes = getattr(self.tensor_meta, "num_uncompressed_bytes", None)
        if nbytes is None:
            nbytes = self._get_num_uncompressed_bytes()
        return nbytes

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
    def commit_diff(self) -> CommitDiff:
        """Gets the commit diff from cache, if one is not found it creates a blank one.

        Returns:
            CommitDiff: The commit diff keeps track of all the changes in the current commit.
        """
        commit_id = self.version_state["commit_id"]
        key = get_tensor_commit_diff_key(self.key, commit_id)
        if not self.commit_diff_exists:
            diff = CommitDiff(first_index=self.num_samples)
            self.meta_cache[key] = diff
            return diff

        diff = self.meta_cache.get_cachable(key, CommitDiff)
        return diff

    @property
    def commit_diff_exists(self) -> bool:
        try:
            commit_id = self.version_state["commit_id"]
            key = get_tensor_commit_diff_key(self.key, commit_id)
            self.meta_cache[key]
            return True
        except KeyError:
            return False

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
    def tile_encoder(self) -> TileEncoder:
        """Gets the tile encoder from cache, if one is not found it creates a blank encoder."""
        commit_id = self.version_state["commit_id"]
        key = get_tensor_tile_encoder_key(self.key, commit_id)
        if not self.tile_encoder_exists:
            enc = TileEncoder()
            self.meta_cache[key] = enc
            return enc

        enc = self.meta_cache.get_cachable(key, TileEncoder)
        return enc

    @property
    def tile_encoder_exists(self) -> bool:
        try:
            commit_id = self.version_state["commit_id"]
            key = get_tensor_tile_encoder_key(self.key, commit_id)
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
        return int(np.uint32(self.chunk_id_encoder.num_samples))

    @property
    def last_chunk_key(self) -> str:
        last_chunk_name = self.last_chunk_name
        commit_id = self.get_chunk_commit(last_chunk_name)
        return get_chunk_key(self.key, last_chunk_name, commit_id)

    @property
    def last_chunk_name(self) -> str:
        return self.chunk_id_encoder.get_name_for_chunk(-1)

    def last_chunk(self) -> Optional[BaseChunk]:
        last_index = self.num_samples - 1
        if self.num_chunks == 0 or last_index in self.tile_encoder:
            return None
        chunk_name = self.last_chunk_name
        chunk_commit_id = self.get_chunk_commit(chunk_name)
        chunk_key = get_chunk_key(self.key, chunk_name, chunk_commit_id)
        chunk = self.get_chunk(chunk_key)
        if chunk_commit_id != self.version_state["commit_id"]:
            chunk = self.copy_chunk_to_new_commit(chunk, chunk_name)
        chunk.key = chunk_key  # type: ignore
        self.add_chunk_to_dirty_keys(chunk)
        return chunk

    def add_chunk_to_dirty_keys(self, chunk: BaseChunk):
        """Adds the chunk to cache if not in dirty keys to ensure persistence."""
        if chunk.key not in self.cache.dirty_keys:  # type: ignore
            self.cache[chunk.key] = chunk  # type: ignore

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
        self.add_cachables_to_cache_dirty_keys()
        # if not the head node, checkout to an auto branch that is newly created
        auto_checkout(self.version_state, self.cache)
        ffw_chunk_id_encoder(self.chunk_id_encoder)

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
        if len(samples) == 0:
            return
        self._write_initialization()
        initial_autoflush = self.cache.autoflush
        self.cache.autoflush = False

        samples = self._sanitize_samples(samples)
        current_chunk = self.last_chunk() or self._create_new_chunk()
        enc = self.chunk_id_encoder
        commit_diff = self.commit_diff
        while len(samples) > 0:
            num_samples_added = current_chunk.extend_if_has_space(samples)
            if num_samples_added == 0:
                current_chunk = self._create_new_chunk()

            elif num_samples_added == PARTIAL_NUM_SAMPLES:
                sample = samples[0]
                if sample.is_first_write:
                    enc.register_samples(1)
                if sample.is_last_write:
                    self.tile_encoder.register_sample(sample, self.num_samples - 1)
                    samples = samples[1:]
                    commit_diff.add_data(1)
                if len(samples) > 0:
                    current_chunk = self._create_new_chunk()
            else:
                num = int(num_samples_added)
                enc.register_samples(num)
                samples = samples[num:]
                commit_diff.add_data(num)

        self.cache.autoflush = initial_autoflush
        self.cache.maybe_flush()

    def add_cachables_to_cache_dirty_keys(self):
        """Adds all the cachables to the cache as dirty keys."""
        if self.cachables_in_dirty_keys:
            return

        commit_id = self.version_state["commit_id"]

        # synchronize tensor meta
        tensor_meta_key = get_tensor_meta_key(self.key, commit_id)
        self.meta_cache[tensor_meta_key] = self.tensor_meta

        # synchronize chunk ID encoder
        chunk_id_key = get_chunk_id_encoder_key(self.key, commit_id)
        self.meta_cache[chunk_id_key] = self.chunk_id_encoder

        # synchronize tile encoder
        tile_encoder_key = get_tensor_tile_encoder_key(self.key, commit_id)
        self.meta_cache[tile_encoder_key] = self.tile_encoder

        # synchronize commit diff
        commit_diff_key = get_tensor_commit_diff_key(self.key, commit_id)
        self.meta_cache[commit_diff_key] = self.commit_diff

        # first commit doesn't have commit chunk set
        if commit_id != FIRST_COMMIT_ID:
            # synchronize current chunk set, all older ones are immutable
            commit_chunk_set_key = get_tensor_commit_chunk_set_key(self.key, commit_id)
            self.meta_cache[commit_chunk_set_key] = self.commit_chunk_set  # type: ignore
        self.cachables_in_dirty_keys = True

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
        initial_autoflush = self.cache.autoflush
        self.cache.autoflush = False

        if operator is not None:
            return self._update_with_operator(index, samples, operator)

        enc = self.chunk_id_encoder
        index_length = index.length(self.num_samples)
        samples = make_sequence(samples, index_length)
        nbytes_after_updates = []
        global_sample_indices = tuple(index.values[0].indices(self.num_samples))
        for i, sample in enumerate(samples):
            global_sample_index = global_sample_indices[i]  # TODO!
            chunks = self.get_chunks_for_sample(global_sample_index, copy=True)
            if len(chunks) > 1:
                raise NotImplementedError(
                    "You can't update a sample that is present in multiple chunks."
                )
            chunk = chunks[0]
            self.add_chunk_to_dirty_keys(chunk)
            local_sample_index = enc.translate_index_relative_to_chunks(
                global_sample_index
            )
            # tensor_meta.update_shape_interval(shape)
            chunk.update_sample(local_sample_index, sample)
            self.commit_diff.update_data(global_sample_index)

            # only care about deltas if it isn't the last chunk
            if chunk.key != self.last_chunk_key:  # type: ignore
                nbytes_after_updates.append(chunk.nbytes)

        self.cache.autoflush = initial_autoflush
        self.cache.maybe_flush()
        chunk_min, chunk_max = self.min_chunk_size, self.max_chunk_size
        check_suboptimal_chunks(nbytes_after_updates, chunk_min, chunk_max)

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
        chunks = self.get_chunks_for_sample(global_sample_index)
        if len(chunks) > 1:
            raise NotImplementedError(
                "read_bytes_for_sample() is not implemented for tiled samples."
            )
        chunk = chunks[0]
        buffer = chunk.memoryview_data
        if not buffer:
            return b""
        local_sample_index = enc.translate_index_relative_to_chunks(global_sample_index)
        sb, eb = chunk.byte_positions_encoder[local_sample_index]
        return buffer[sb:eb].tobytes()

    def read_shape_for_sample(self, global_sample_index: int) -> Tuple[int, ...]:
        enc = self.chunk_id_encoder
        chunks = self.get_chunks_for_sample(global_sample_index)
        if len(chunks) == 1:
            local_sample_index = enc.translate_index_relative_to_chunks(
                global_sample_index
            )
            return tuple(map(int, chunks[0].shapes_encoder[local_sample_index]))
        else:
            return self.tile_encoder.get_sample_shape(global_sample_index)

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
        enc = self.chunk_id_encoder

        for global_sample_index in index.values[0].indices(length):
            chunks = self.get_chunks_for_sample(global_sample_index)

            if len(chunks) == 1:
                chunk = chunks[0]
                idx = enc.translate_index_relative_to_chunks(global_sample_index)
                sample = chunk.read_sample(idx)
            else:
                tile_enc = self.tile_encoder
                sample = combine_chunks(chunks, global_sample_index, tile_enc)
            shape = sample.shape
            check_sample_shape(shape, last_shape, self.key, index, aslist)
            samples.append(sample)
            last_shape = shape
        return format_read_samples(samples, index, aslist)

    def get_chunks_for_sample(
        self, global_sample_index: int, copy: bool = False
    ) -> List[BaseChunk]:
        """Retrives the `Chunk` object corresponding to `global_sample_index`.
        Args:
            global_sample_index (int): Index relative to the entire tensor representing the sample.
            copy (bool): If True and the chunk exists in a different commit to the current commit, it will be copied. Defaults to False.
        Returns:
            BaseChunk: BaseChunk object that contains `global_sample_index`.
        """
        enc = self.chunk_id_encoder
        chunk_ids = enc[global_sample_index]
        chunk_list = []
        for chunk_id in chunk_ids:
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
            chunk_list.append(chunk)
        return chunk_list

    def validate_num_samples_is_synchronized(self):
        """Check if tensor meta length and chunk ID encoder are representing the same number of samples.
        Helpful for determining if a user has tampered with the tensor meta or the chunk ID encoder, or if
        the tensor was corruptd.

        Raises:
            CorruptedMetaError: tensor_meta and chunk_id_encoder must have the same num samples.
        """

        tensor_meta_length = self.tensor_meta.length

        # compare chunk ID encoder and tensor meta
        chunk_id_num_samples = np.uint32(
            self.chunk_id_encoder.num_samples if self.chunk_id_encoder_exists else 0
        )
        if tensor_meta_length != chunk_id_num_samples:
            commit_id = self.version_state["commit_id"]
            tkey = get_tensor_meta_key(self.key, commit_id)
            ikey = get_chunk_id_encoder_key(self.key, commit_id)
            raise CorruptedMetaError(
                f"'{tkey}' and '{ikey}' have a record of different numbers of samples. Got {tensor_meta_length} and {chunk_id_num_samples} respectively."
            )

    def _get_all_chunks(self):
        n_samples = self.num_samples
        chunk_names = self.get_chunk_names_for_multiple_indexes(0, n_samples, n_samples)
        commit_id = self.version_state["commit_id"]
        chunk_keys = [get_chunk_key(self.key, name, commit_id) for name in chunk_names]
        chunks = [self.cache.get_cachable(key, Chunk) for key in chunk_keys]
        return chunks

    def _get_num_compressed_bytes(self):
        if self.num_chunks == 0:
            return None

        chunks = self._get_all_chunks()
        nbytes = 0

        for chunk in chunks:
            nbytes += chunk.num_data_bytes

        return nbytes

    def _get_chunk_uncompressed_size(self, chunk):
        dtype = self.tensor_meta.dtype
        if not dtype:
            return
        itemsize = np.dtype(dtype).itemsize
        nbytes = 0

        shapes = [
            chunk.shapes_encoder[i] for i in range(chunk.shapes_encoder.num_samples)
        ]
        nbytes += sum([np.prod(shape).item() for shape in shapes]) * itemsize
        return nbytes

    def _get_num_uncompressed_bytes(self):
        if self.num_chunks == 0:
            return None

        chunks = self._get_all_chunks()
        nbytes = 0

        for chunk in chunks:
            nbytes += self._get_chunk_uncompressed_size(chunk)

        return nbytes
