import hub
import numpy as np
from typing import Any, Callable, Dict, Optional, Sequence, Union, List, Tuple
from hub.api.info import Info
from hub.core.tensor_link import get_link_transform
from hub.core.version_control.commit_diff import CommitDiff
from hub.core.version_control.commit_node import CommitNode  # type: ignore
from hub.core.version_control.commit_chunk_set import CommitChunkSet  # type: ignore
from typing import Any, Dict, List, Optional, Sequence, Union, Callable
from hub.core.meta.encode.tile import TileEncoder
from hub.core.storage.provider import StorageProvider
from hub.core.storage import S3Provider, GCSProvider
from hub.core.tiling.deserialize import combine_chunks, translate_slices, coalesce_tiles
from hub.core.tiling.serialize import break_into_tiles
from hub.util.casting import get_empty_sample, intelligent_cast
from hub.util.shape_interval import ShapeInterval
from hub.constants import DEFAULT_MAX_CHUNK_SIZE, FIRST_COMMIT_ID, PARTIAL_NUM_SAMPLES
from hub.core.chunk.base_chunk import BaseChunk, InputSample
from hub.core.chunk.chunk_compressed_chunk import ChunkCompressedChunk
from hub.core.chunk.sample_compressed_chunk import SampleCompressedChunk
from hub.core.chunk.uncompressed_chunk import UncompressedChunk
from hub.core.fast_forwarding import ffw_chunk_id_encoder
from hub.core.index.index import Index, IndexEntry
from hub.core.meta.encode.chunk_id import CHUNK_ID_COLUMN, ChunkIdEncoder
from hub.core.meta.encode.sequence import SequenceEncoder
from hub.core.meta.tensor_meta import TensorMeta
from hub.core.storage.lru_cache import LRUCache
from hub.util.casting import get_dtype, get_htype
from hub.util.chunk_engine import (
    check_samples_type,
    make_sequence,
    check_suboptimal_chunks,
    check_sample_shape,
)
from hub.util.keys import (
    get_chunk_id_encoder_key,
    get_sequence_encoder_key,
    get_tensor_commit_diff_key,
    get_tensor_meta_key,
    get_chunk_key,
    get_tensor_commit_chunk_set_key,
    get_tensor_meta_key,
    get_tensor_tile_encoder_key,
    get_tensor_info_key,
)
from hub.util.exceptions import (
    CorruptedMetaError,
    DynamicTensorNumpyError,
    ReadOnlyModeError,
)
from hub.util.remove_cache import get_base_storage
from hub.compression import VIDEO_COMPRESSIONS
from itertools import chain, repeat
from collections.abc import Iterable


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
        self.base_storage = get_base_storage(cache)
        self._meta_cache = meta_cache
        self.version_state = version_state
        self.compression = None
        self.chunk_class = BaseChunk

        self._tensor_meta: Optional[TensorMeta] = None
        self._tensor_meta_commit_id: Optional[str] = None

        self._chunk_id_encoder: Optional[ChunkIdEncoder] = None
        self._chunk_id_encoder_commit_id: Optional[str] = None

        self._sequence_encoder: Optional[SequenceEncoder] = None
        self._sequence_encoder_commit_id: Optional[str] = None

        self._tile_encoder: Optional[TileEncoder] = None
        self._tile_encoder_commit_id: Optional[str] = None

        self._commit_chunk_set: Optional[CommitChunkSet] = None
        self._commit_chunk_set_commit_id: Optional[str] = None

        self._commit_diff: Optional[CommitDiff] = None
        self._commit_diff_commit_id: Optional[str] = None

        self._active_appended_chunk: Optional[BaseChunk] = None
        self._active_updated_chunk: Optional[BaseChunk] = None

        self._info: Optional[Info] = None
        self._info_commit_id: Optional[str] = None

        self._all_chunk_engines: Optional[Dict[str, ChunkEngine]] = None

        tensor_meta = self.tensor_meta

        if tensor_meta.sample_compression:
            self.compression = tensor_meta.sample_compression
            self.chunk_class = SampleCompressedChunk
        elif tensor_meta.chunk_compression:
            self.compression = tensor_meta.chunk_compression
            self.chunk_class = ChunkCompressedChunk
        else:
            self.chunk_class = UncompressedChunk

        self.cached_data: Optional[np.ndarray] = None
        self.cache_range: range = range(0)

    @property
    def is_data_cachable(self):
        tensor_meta = self.tensor_meta
        return (
            self.chunk_class == UncompressedChunk
            and tensor_meta.htype not in ["text", "json", "list"]
            and (tensor_meta.max_shape == tensor_meta.min_shape)
            and (np.prod(tensor_meta.max_shape) < 20)
        )

    @property
    def commit_id(self):
        return self.version_state["commit_id"]

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
        commit_id = self.commit_id
        if self._tensor_meta is None or self._tensor_meta_commit_id != commit_id:
            key = get_tensor_meta_key(self.key, commit_id)
            self._tensor_meta = self.meta_cache.get_hub_object(key, TensorMeta)
            self._tensor_meta_commit_id = commit_id
            self.meta_cache.register_hub_object(key, self._tensor_meta)
        return self._tensor_meta

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
        commit_id = self.commit_id
        if (
            self._chunk_id_encoder is None
            or self._chunk_id_encoder_commit_id != commit_id
        ):
            commit_id = self.commit_id
            key = get_chunk_id_encoder_key(self.key, commit_id)
            if not self.chunk_id_encoder_exists:
                enc = ChunkIdEncoder()
                try:
                    self.meta_cache[key] = enc
                except ReadOnlyModeError:
                    pass
            else:
                enc = self.meta_cache.get_hub_object(key, ChunkIdEncoder)
            self._chunk_id_encoder = enc
            self._chunk_id_encoder_commit_id = commit_id
            self.meta_cache.register_hub_object(key, enc)
        return self._chunk_id_encoder

    @property
    def commit_chunk_set(self) -> Optional[CommitChunkSet]:
        """Gets the commit chunk set from cache, if one is not found it creates a blank one.

        Returns:
            Optional[CommitChunkSet]: The commit chunk set keeps track of all the chunks present in the current commit, returns None for the first commit.
        """
        commit_id = self.commit_id
        if commit_id == FIRST_COMMIT_ID:
            # the first commit doesn't need a commit chunk set
            return None
        if (
            self._commit_chunk_set is None
            or self._commit_chunk_set_commit_id != commit_id
        ):
            key = get_tensor_commit_chunk_set_key(self.key, commit_id)
            if not self.commit_chunk_set_exists:
                cset = CommitChunkSet()
                try:
                    self.meta_cache[key] = cset
                except ReadOnlyModeError:
                    pass
            else:
                cset = self.meta_cache.get_hub_object(key, CommitChunkSet)
            self._commit_chunk_set = cset
            self._commit_chunk_set_commit_id = commit_id
            self.meta_cache.register_hub_object(key, cset)
        return self._commit_chunk_set

    @property
    def commit_chunk_set_exists(self) -> bool:
        """Checks if the commit chunk set exists for the given tensor in the current commit."""
        commit_id = self.commit_id
        if (
            self._commit_chunk_set is not None
            and self._commit_chunk_set_commit_id == commit_id
        ):
            return True

        try:
            key = get_tensor_commit_chunk_set_key(self.key, commit_id)
            self.meta_cache[key]
            return True
        except KeyError:
            return False

    @property
    def commit_diff(self) -> CommitDiff:
        """Gets the commit diff from cache, if one is not found it creates a blank one.

        Returns:
            CommitDiff: The commit diff keeps track of all the changes in the current commit.
        """
        commit_id = self.commit_id
        if self._commit_diff is None or self._commit_diff_commit_id != commit_id:
            key = get_tensor_commit_diff_key(self.key, commit_id)
            if not self.commit_diff_exists:
                diff = CommitDiff(self.num_samples)
                try:
                    self.meta_cache[key] = diff
                except ReadOnlyModeError:
                    pass
            else:
                diff = self.meta_cache.get_hub_object(key, CommitDiff)
            self._commit_diff = diff
            self._commit_diff_commit_id = commit_id
            self.meta_cache.register_hub_object(key, diff)
        return self._commit_diff

    @property
    def commit_diff_exists(self) -> bool:
        commit_id = self.commit_id
        if self._commit_diff is not None and self._commit_diff_commit_id == commit_id:
            return True
        try:
            key = get_tensor_commit_diff_key(self.key, commit_id)
            self.meta_cache[key]
            return True
        except KeyError:
            return False

    @property
    def chunk_id_encoder_exists(self) -> bool:
        commit_id = self.commit_id
        if (
            self._chunk_id_encoder is not None
            and self._chunk_id_encoder_commit_id == commit_id
        ):
            return True
        try:
            key = get_chunk_id_encoder_key(self.key, commit_id)
            self.meta_cache[key]
            return True
        except KeyError:
            return False

    def _is_tiled_sample(self, global_sample_index):
        return global_sample_index in self.tile_encoder

    @property
    def tile_encoder(self) -> TileEncoder:
        """Gets the tile encoder from cache, if one is not found it creates a blank encoder."""
        commit_id = self.commit_id
        if self._tile_encoder is None or self._tile_encoder_commit_id != commit_id:
            key = get_tensor_tile_encoder_key(self.key, commit_id)
            if not self.tile_encoder_exists:
                enc = TileEncoder()
                try:
                    self.meta_cache[key] = enc
                except ReadOnlyModeError:
                    pass
            else:
                enc = self.meta_cache.get_hub_object(key, TileEncoder)
            self._tile_encoder = enc
            self._tile_encoder_commit_id = commit_id
            self.meta_cache.register_hub_object(key, enc)
        return self._tile_encoder

    @property
    def tile_encoder_exists(self) -> bool:
        commit_id = self.commit_id
        if self._tile_encoder is not None and self._tile_encoder_commit_id == commit_id:
            return True

        try:
            key = get_tensor_tile_encoder_key(self.key, commit_id)
            self.meta_cache[key]
            return True
        except KeyError:
            return False

    @property
    def creds_encoder(self):
        return None

    @property
    def num_chunks(self) -> int:
        if not self.chunk_id_encoder_exists:
            return 0
        return self.chunk_id_encoder.num_chunks

    @property
    def num_samples(self) -> int:
        return self.tensor_meta.length

    @property
    def last_chunk_key(self) -> str:
        last_chunk_name = self.last_appended_chunk_name
        commit_id = self.get_chunk_commit(last_chunk_name)
        return get_chunk_key(self.key, last_chunk_name, commit_id)

    def get_chunk_key_for_id(self, chunk_id) -> str:
        chunk_name = ChunkIdEncoder.name_from_id(chunk_id)
        commit_id = self.get_chunk_commit(chunk_name)
        return get_chunk_key(self.key, chunk_name, commit_id)

    @property
    def active_appended_chunk(self):
        return self._active_appended_chunk

    @active_appended_chunk.setter
    def active_appended_chunk(self, value):
        if self.active_appended_chunk is not None:
            self.cache.remove_hub_object(self.active_appended_chunk.key)
        self._active_appended_chunk = value
        if value is not None:
            self.cache.register_hub_object(value.key, value)

    @property
    def active_updated_chunk(self):
        return self._active_updated_chunk

    @active_updated_chunk.setter
    def active_updated_chunk(self, value):
        if self.active_updated_chunk is not None:
            self.cache.remove_hub_object(self.active_updated_chunk.key)
        self._active_updated_chunk = value
        if value is not None:
            self.cache.register_hub_object(value.key, value)

    @property
    def last_appended_chunk_name(self) -> str:
        return self.chunk_id_encoder.get_name_for_chunk(-1)

    @property
    def last_appended_chunk_id(self) -> str:
        return self.chunk_id_encoder.get_id_for_chunk(-1)

    def last_appended_chunk(self) -> Optional[BaseChunk]:
        last_index = self.num_samples - 1
        if self.num_chunks == 0 or last_index in self.tile_encoder:
            return None
        chunk_name = self.last_appended_chunk_name
        chunk_commit_id = self.get_chunk_commit(chunk_name)
        chunk_key = get_chunk_key(self.key, chunk_name, chunk_commit_id)
        chunk = self.get_chunk(chunk_key)
        chunk.key = chunk_key  # type: ignore
        chunk.id = self.last_appended_chunk_id  # type: ignore
        if chunk_commit_id != self.commit_id:
            chunk = self.copy_chunk_to_new_commit(chunk, chunk_name)
        self.active_appended_chunk = chunk
        return chunk

    def get_chunk(self, chunk_key: str) -> BaseChunk:
        return self.cache.get_hub_object(chunk_key, self.chunk_class, self.chunk_args)

    def get_chunk_from_chunk_id(self, chunk_id, copy: bool = False) -> BaseChunk:
        chunk_name = ChunkIdEncoder.name_from_id(chunk_id)
        chunk_commit_id = self.get_chunk_commit(chunk_name)
        chunk_key = get_chunk_key(self.key, chunk_name, chunk_commit_id)
        chunk = self.get_chunk(chunk_key)
        chunk.key = chunk_key  # type: ignore
        chunk.id = chunk_id  # type: ignore
        if copy and chunk_commit_id != self.commit_id:
            chunk = self.copy_chunk_to_new_commit(chunk, chunk_name)
        return chunk

    def get_video_chunk(self, chunk_id, copy: bool = False):
        """Returns video chunks. Chunk will contain presigned url to the video instead of data if the chunk is large."""
        chunk_name = ChunkIdEncoder.name_from_id(chunk_id)
        chunk_commit_id = self.get_chunk_commit(chunk_name)
        chunk_key = get_chunk_key(self.key, chunk_name, chunk_commit_id)

        base_storage = self.base_storage
        stream = False
        if isinstance(base_storage, (S3Provider, GCSProvider)):
            chunk_size = base_storage.get_object_size(chunk_key)
            stream = chunk_size > self.min_chunk_size
            if stream:
                chunk = self.cache.get_hub_object(
                    chunk_key, self.chunk_class, meta=self.chunk_args, url=True
                )
        if not stream:
            chunk = self.cache.get_hub_object(
                chunk_key, self.chunk_class, meta=self.chunk_args
            )
        chunk.key = chunk_key  # type: ignore
        chunk.id = chunk_id  # type: ignore
        if copy and chunk_commit_id != self.commit_id:
            chunk = self.copy_chunk_to_new_commit(chunk, chunk_name)
        return chunk, stream

    def copy_chunk_to_new_commit(self, chunk, chunk_name):
        """Copies the chunk to the current commit.

        Returns the copied chunk.
        """
        new_chunk_key = get_chunk_key(self.key, chunk_name, self.commit_id)
        chunk_id = chunk.id
        chunk = chunk.copy(self.chunk_args)
        chunk.key = new_chunk_key
        chunk.id = chunk_id
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
                    chunk_set = self.meta_cache.get_hub_object(
                        chunk_set_key, CommitChunkSet
                    ).chunks
            except Exception:
                commit_chunk_set = CommitChunkSet()
                try:
                    self.meta_cache[chunk_set_key] = commit_chunk_set
                except ReadOnlyModeError:
                    # put CommitChunkSet in hub_objects to keep in cache temporarily, but won't write to storage
                    # this shouldn't happen in latest version of hub, chunk set would always be present
                    self.meta_cache.hub_objects[chunk_set_key] = commit_chunk_set
                chunk_set = set()
            if chunk_name in chunk_set:
                return commit_id
            cur_node = cur_node.parent  # type: ignore
        # the first commit doesn't have a commit chunk set, so any chunk that wasn't found belongs to the first commit
        return FIRST_COMMIT_ID

    def _write_initialization(self):
        ffw_chunk_id_encoder(self.chunk_id_encoder)

    def _convert_to_list(self, samples):
        if self.chunk_class != UncompressedChunk:
            return True
        elif isinstance(samples, np.ndarray):
            return samples[0].nbytes >= self.min_chunk_size
        return True

    def check_each_sample(self, samples):
        return

    def _sanitize_samples(self, samples):
        check_samples_type(samples)
        verified_samples = self.check_each_sample(samples)
        tensor_meta = self.tensor_meta
        if tensor_meta.htype is None:
            tensor_meta.set_htype(get_htype(samples))
        if tensor_meta.dtype is None:
            tensor_meta.set_dtype(get_dtype(samples))
        if self._convert_to_list(samples):
            samples = list(samples)
        return samples, verified_samples

    def _samples_to_chunks(
        self,
        samples,
        start_chunk: Optional[BaseChunk] = None,
        register: bool = True,
        update_commit_diff: bool = False,
    ):
        current_chunk = start_chunk
        updated_chunks = []
        if current_chunk is None:
            current_chunk = self._create_new_chunk(register)
            updated_chunks.append(current_chunk)
        enc = self.chunk_id_encoder
        tiles = {}
        nsamples = len(samples)
        if register and update_commit_diff:
            commit_diff = self.commit_diff
        while len(samples) > 0:
            num_samples_added = current_chunk.extend_if_has_space(samples)  # type: ignore
            self.register_new_creds(num_samples_added, samples)
            if num_samples_added == 0:
                current_chunk = self._create_new_chunk(register)
                updated_chunks.append(current_chunk)
            elif num_samples_added == PARTIAL_NUM_SAMPLES:
                sample = samples[0]
                if register and sample.is_first_write:
                    enc.register_samples(1)
                if sample.is_last_write:
                    if register:
                        self.tile_encoder.register_sample(sample, self.num_samples - 1)
                        if update_commit_diff:
                            commit_diff.add_data(1)
                    else:
                        tiles[nsamples - len(samples)] = (
                            sample.sample_shape,
                            sample.tile_shape,
                        )
                    samples = samples[1:]
                if len(samples) > 0:
                    current_chunk = self._create_new_chunk(register)
                    updated_chunks.append(current_chunk)
            else:
                if not updated_chunks:
                    updated_chunks.append(current_chunk)
                num = int(num_samples_added)
                if register:
                    enc.register_samples(num)
                    if update_commit_diff:
                        commit_diff.add_data(num)
                samples = samples[num:]
        if register:
            return updated_chunks
        return updated_chunks, tiles

    def register_new_creds(self, num_samples_added, samples):
        return

    def update_creds(self, sample_index, sample):
        return

    def _extend(self, samples, update_commit_diff=True):
        if isinstance(samples, hub.Tensor):
            for sample in samples:
                self._extend(
                    [sample], update_commit_diff=update_commit_diff
                )  # TODO optimize this
            return
        if len(samples) == 0:
            return
        samples, verified_samples = self._sanitize_samples(samples)
        self._samples_to_chunks(
            samples,
            start_chunk=self.last_appended_chunk(),
            register=True,
            update_commit_diff=update_commit_diff,
        )
        return verified_samples

    def extend(self, samples, link_callback: Optional[Callable] = None):
        self._write_initialization()
        initial_autoflush = self.cache.autoflush
        self.cache.autoflush = False

        if self.is_sequence:
            for sample in samples:
                verified_sample = self._extend(sample, update_commit_diff=False)
                self.sequence_encoder.register_samples(len(sample), 1)
                self.commit_diff.add_data(1)

                ls = verified_sample or sample
                if link_callback:
                    link_callback(ls, flat=False)
                    for s in ls:
                        link_callback(s, flat=True)
        else:
            verified_samples = self._extend(samples)
            ls = verified_samples or samples
            if link_callback:
                for sample in ls:
                    link_callback(sample, flat=None)

        self.cache.autoflush = initial_autoflush
        self.cache.maybe_flush()

    def _create_new_chunk(self, register=True) -> BaseChunk:
        """Creates and returns a new `Chunk`. Automatically creates an ID for it and puts a reference in the cache."""
        chunk_id = self.chunk_id_encoder.generate_chunk_id(register=register)
        chunk = self.chunk_class(*self.chunk_args)  # type: ignore
        chunk_name = ChunkIdEncoder.name_from_id(chunk_id)
        chunk_key = get_chunk_key(self.key, chunk_name, self.commit_id)
        if self.commit_chunk_set is not None:
            self.commit_chunk_set.add(chunk_name)
        chunk.key = chunk_key  # type: ignore
        chunk.id = chunk_id  # type: ignore
        chunk._update_tensor_meta_length = register
        if self.active_appended_chunk is not None:
            self.write_chunk_to_storage(self.active_appended_chunk)
        self.active_appended_chunk = chunk
        return chunk

    def clear(self):
        """Clears all samples and cachables."""
        self.cache.check_readonly()

        commit_id = self.commit_id

        chunk_folder_path = get_chunk_key(self.key, "", commit_id)
        self.cache.clear(prefix=chunk_folder_path)

        enc_key = get_chunk_id_encoder_key(self.key, commit_id)
        self._chunk_id_encoder = None
        try:
            del self.meta_cache[enc_key]
        except KeyError:
            pass

        info_key = get_tensor_info_key(self.key, commit_id)
        try:
            self._info = None
            del self.cache[info_key]
        except KeyError:
            pass

        self.commit_diff.clear_data()

        tile_encoder_key = get_tensor_tile_encoder_key(self.key, commit_id)
        try:
            self._tile_encoder = None
            del self.cache[tile_encoder_key]
        except KeyError:
            pass

        seq_encoder_key = get_sequence_encoder_key(self.key, commit_id)
        try:
            self._sequence_encoder = None
            del self.cache[seq_encoder_key]
        except KeyError:
            pass

        self.tensor_meta.length = 0
        self.tensor_meta.min_shape = []
        self.tensor_meta.max_shape = []
        self.tensor_meta.is_dirty = True

        self.cache.maybe_flush()
        self.meta_cache.maybe_flush()

    def _replace_tiled_sample(self, global_sample_index: int, sample):
        new_chunks, tiles = self._samples_to_chunks(
            [sample], start_chunk=None, register=False
        )
        new_chunk_ids = [chunk.id for chunk in new_chunks]
        self.chunk_id_encoder._replace_chunks_for_tiled_sample(
            global_sample_index, new_chunk_ids
        )
        if tiles:
            self.tile_encoder.entries[global_sample_index] = tiles[0]
        else:
            del self.tile_encoder.entries[global_sample_index]

    def _update_tiled_sample(self, global_sample_index: int, index: Index, sample):
        if len(index.values) == 1:
            self._replace_tiled_sample(global_sample_index, sample)
            return
        enc = self.chunk_id_encoder
        tile_enc = self.tile_encoder
        chunk_ids = enc[global_sample_index]
        sample_shape = tile_enc.get_sample_shape(global_sample_index)
        tile_shape = tile_enc.get_tile_shape(global_sample_index)
        ordered_tile_ids = np.array(chunk_ids).reshape(
            tile_enc.get_tile_layout_shape(global_sample_index)
        )
        tiles_index, sample_index = translate_slices(
            [v.value for v in index.values[1:]], sample_shape, tile_shape  # type: ignore
        )
        required_tile_ids = ordered_tile_ids[tiles_index]
        tiles = np.vectorize(
            lambda chunk_id: self.get_chunk_from_chunk_id(
                chunk_id, copy=True
            ).read_sample(0),
            otypes=[object],
        )(required_tile_ids)
        current_sample = coalesce_tiles(tiles, tile_shape, None, self.tensor_meta.dtype)
        new_sample = current_sample
        new_sample[sample_index] = sample
        new_tiles = break_into_tiles(
            new_sample, tile_enc.get_tile_shape(global_sample_index)
        )
        chunk_ids = required_tile_ids
        for chunk_id, tile in zip(chunk_ids.reshape(-1), new_tiles.reshape(-1)):
            chunk = self.get_chunk_from_chunk_id(int(chunk_id), copy=True)
            curr_shape = chunk.shapes_encoder[-1]
            assert curr_shape == tile.shape, (curr_shape, tile.shape)
            chunk.update_sample(0, tile)
            if (
                self.active_updated_chunk is not None
                and self.active_updated_chunk.key != chunk.key  # type: ignore
            ):
                self.write_chunk_to_storage(self.active_updated_chunk)
            self.active_updated_chunk = chunk

    def pad_and_append(
        self,
        num_samples_to_pad: int,
        value,
        append_link_callback=None,
        update_link_callback=None,
    ):
        """Pads the tensor with empty samples and appends value at the end."""
        update_first_sample = False
        if num_samples_to_pad > 0:
            if self.num_samples == 0:
                # set htype, dtype, shape, we later update it with empty sample
                self.extend([value], link_callback=append_link_callback)
                num_samples_to_pad -= 1
                update_first_sample = True

            htype = self.tensor_meta.htype
            if htype in ["json", "text", "list"]:
                empty_sample = get_empty_sample(htype)
                empty_samples = [empty_sample] * num_samples_to_pad
            else:
                ndim = len(self.tensor_meta.max_shape)
                shape = tuple([num_samples_to_pad] + [0] * ndim)
                dtype = self.tensor_meta.dtype
                empty_sample = np.zeros(shape[1:], dtype=dtype)
                empty_samples = np.zeros(shape, dtype=dtype)  # type: ignore

            if update_first_sample:
                self.update(Index(0), empty_sample, link_callback=update_link_callback)

            # pad
            self.extend(empty_samples, link_callback=append_link_callback)

        self.extend([value], link_callback=append_link_callback)

    def update(
        self,
        index: Index,
        samples: Union[np.ndarray, Sequence[InputSample], InputSample],
        operator: Optional[str] = None,
        link_callback: Optional[Callable] = None,
    ):
        """Update data at `index` with `samples`."""
        (self._sequence_update if self.is_sequence else self._update)(  # type: ignore
            index,
            samples,
            operator,
            link_callback=link_callback,
        )

    def _update(
        self,
        index: Index,
        samples: Union[np.ndarray, Sequence[InputSample], InputSample],
        operator: Optional[str] = None,
        update_commit_diff: bool = True,
        link_callback: Optional[Callable] = None,
    ):
        """Update data at `index` with `samples`."""
        self._write_initialization()
        self.cached_data = None
        initial_autoflush = self.cache.autoflush
        self.cache.autoflush = False

        if operator is not None:
            return self._update_with_operator(index, samples, operator)

        enc = self.chunk_id_encoder
        index_length = index.length(self.num_samples)
        samples = make_sequence(samples, index_length)
        verified_samples = self.check_each_sample(samples)
        nbytes_after_updates = []
        global_sample_indices = tuple(index.values[0].indices(self.num_samples))
        is_sequence = self.is_sequence
        for i, sample in enumerate(samples):
            global_sample_index = global_sample_indices[i]  # TODO!
            if self._is_tiled_sample(global_sample_index):
                self._update_tiled_sample(global_sample_index, index, sample)
            else:
                chunk = self.get_chunks_for_sample(global_sample_index, copy=True)[0]
                local_sample_index = enc.translate_index_relative_to_chunks(
                    global_sample_index
                )
                chunk.update_sample(local_sample_index, sample)
                if (
                    self.active_updated_chunk is not None
                    and self.active_updated_chunk.key != chunk.key  # type: ignore
                ):
                    self.write_chunk_to_storage(self.active_updated_chunk)
                self.active_updated_chunk = chunk

                # only care about deltas if it isn't the last chunk
                if chunk.key != self.last_chunk_key:  # type: ignore
                    nbytes_after_updates.append(chunk.nbytes)
            self.update_creds(global_sample_index, sample)
            if update_commit_diff:
                self.commit_diff.update_data(global_sample_index)
            chunk_min, chunk_max = self.min_chunk_size, self.max_chunk_size
            check_suboptimal_chunks(nbytes_after_updates, chunk_min, chunk_max)
            if link_callback:
                new_sample = verified_samples[i] if verified_samples else sample
                link_callback(
                    global_sample_index,
                    sub_index=Index(index.values[1:]),
                    new_sample=new_sample,
                    flat=True if is_sequence else None,
                )

        self.cache.autoflush = initial_autoflush
        self.cache.maybe_flush()
        return verified_samples

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
            if len(index) > 1:
                index1 = Index(index.values[:1])
                index2 = Index(index.values[1:])
            else:
                index1 = index
                index2 = None
            arr = self._numpy(index1, use_data_cache=False)
            view = arr
            if index2:
                for v in index2.values:
                    view = view[v.value]  # type: ignore
        except DynamicTensorNumpyError:
            raise NotImplementedError(
                "Inplace update operations are not available for dynamic tensors yet."
            )
        tensor_meta = self.tensor_meta

        dt, ht = tensor_meta.dtype, tensor_meta.htype
        samples = intelligent_cast(samples, dt, ht)
        getattr(view, operator)(samples)
        self._update(index, arr)

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

    def read_shape_for_sample(
        self,
        global_sample_index: int,
    ) -> Tuple[int, ...]:
        enc = self.chunk_id_encoder
        if self.compression in VIDEO_COMPRESSIONS or self.tensor_meta.htype == "video":
            chunks = [
                self.get_video_chunk(idx)[0]
                for idx in self.chunk_id_encoder[global_sample_index]
            ]
        else:
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
        decompress: bool = True,
    ) -> np.ndarray:
        enc = self.chunk_id_encoder
        local_sample_index = enc.translate_index_relative_to_chunks(global_sample_index)
        return chunk.read_sample(
            local_sample_index, cast=cast, copy=copy, decompress=decompress
        )

    def numpy(
        self, index: Index, aslist: bool = False, use_data_cache: bool = True
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """Reads samples from chunks and returns as a numpy array. If `aslist=True`, returns a sequence of numpy arrays.

        Args:
            index (Index): Represents the samples to read from chunks. See `Index` for more information.
            aslist (bool): If True, the samples will be returned as a list of numpy arrays. If False, returns a single numpy array. Defaults to False.
            use_data_cache (bool): If True, the data cache is used to speed up the read if possible. If False, the data cache is ignored. Defaults to True.

        Raises:
            DynamicTensorNumpyError: If shapes of the samples being read are not all the same.

        Returns:
            Union[np.ndarray, List[np.ndarray]]: Either a list of numpy arrays or a single numpy array (depending on the `aslist` argument).
        """
        return (self._sequence_numpy if self.is_sequence else self._numpy)(
            index, aslist, use_data_cache
        )

    def get_video_sample(self, global_sample_index, index):
        enc = self.chunk_id_encoder
        chunk_ids = enc[global_sample_index]
        local_sample_index = enc.translate_index_relative_to_chunks(global_sample_index)
        chunk, stream = self.get_video_chunk(chunk_ids[0])
        sub_index = index.values[1].value if len(index.values) > 1 else None  # type: ignore
        sample = chunk.read_sample(
            local_sample_index,
            sub_index=sub_index,
            stream=stream,
        )[tuple(entry.value for entry in index.values[2:])]
        return sample

    def get_basic_sample(self, global_sample_index, index):
        enc = self.chunk_id_encoder
        chunk_ids = enc[global_sample_index]
        local_sample_index = enc.translate_index_relative_to_chunks(global_sample_index)
        chunk = self.get_chunk_from_chunk_id(chunk_ids[0])
        return chunk.read_sample(
            local_sample_index, cast=self.tensor_meta.htype != "dicom"
        )[tuple(entry.value for entry in index.values[1:])]

    def get_non_tiled_sample(self, global_sample_index, index):
        if self.compression in VIDEO_COMPRESSIONS or self.tensor_meta.htype == "video":
            return self.get_video_sample(global_sample_index, index)
        return self.get_basic_sample(global_sample_index, index)

    def get_full_tiled_sample(self, global_sample_index):
        chunks = self.get_chunks_for_sample(global_sample_index)
        return combine_chunks(chunks, global_sample_index, self.tile_encoder)

    def get_partial_tiled_sample(self, global_sample_index, index):
        tile_enc = self.tile_encoder
        chunk_ids = self.chunk_id_encoder[global_sample_index]
        sample_shape = tile_enc.get_sample_shape(global_sample_index)
        tile_shape = tile_enc.get_tile_shape(global_sample_index)
        ordered_tile_ids = np.array(chunk_ids).reshape(
            tile_enc.get_tile_layout_shape(global_sample_index)
        )
        tiles_index, sample_index = translate_slices(
            [v.value for v in index.values[1:]], sample_shape, tile_shape  # type: ignore
        )
        required_tile_ids = ordered_tile_ids[tiles_index]
        tiles = np.vectorize(
            lambda chunk_id: self.get_chunk_from_chunk_id(chunk_id).read_sample(0),
            otypes=[object],
        )(required_tile_ids)
        sample = coalesce_tiles(tiles, tile_shape, None, self.tensor_meta.dtype)
        sample = sample[sample_index]
        return sample

    def get_single_sample(self, global_sample_index, index):
        if not self._is_tiled_sample(global_sample_index):
            sample = self.get_non_tiled_sample(global_sample_index, index)
        elif len(index.values) == 1:
            sample = self.get_full_tiled_sample(global_sample_index)
        else:
            sample = self.get_partial_tiled_sample(global_sample_index, index)

        return sample

    def _numpy(
        self, index: Index, aslist: bool = False, use_data_cache: bool = True
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """Reads samples from chunks and returns as a numpy array. If `aslist=True`, returns a sequence of numpy arrays.

        Args:
            index (Index): Represents the samples to read from chunks. See `Index` for more information.
            aslist (bool): If True, the samples will be returned as a list of numpy arrays. If False, returns a single numpy array. Defaults to False.
            use_data_cache (bool): If True, the data cache is used to speed up the read if possible. If False, the data cache is ignored. Defaults to True.

        Raises:
            DynamicTensorNumpyError: If shapes of the samples being read are not all the same.

        Returns:
            Union[np.ndarray, List[np.ndarray]]: Either a list of numpy arrays or a single numpy array (depending on the `aslist` argument).
        """
        length = self.num_samples
        last_shape = None
        if use_data_cache and self.is_data_cachable:
            samples = self.numpy_from_data_cache(index, length, aslist)
        else:
            samples = []
            for global_sample_index in index.values[0].indices(length):
                sample = self.get_single_sample(global_sample_index, index)
                samples.append(sample)
                check_sample_shape(sample.shape, last_shape, self.key, index, aslist)
                last_shape = sample.shape

        if aslist and all(map(np.isscalar, samples)):
            samples = list(arr.item() for arr in samples)

        if not index.values[0].subscriptable():
            samples = samples[0]

        if aslist:
            return samples
        return np.array(samples)

    def numpy_from_data_cache(self, index, length, aslist):
        samples = []
        enc = self.chunk_id_encoder
        for global_sample_index in index.values[0].indices(length):
            if self.cached_data is None or global_sample_index not in self.cache_range:
                row = enc.__getitem__(global_sample_index, True)[0][1]
                chunks = self.get_chunks_for_sample(global_sample_index)
                assert len(chunks) == 1

                chunk = chunks[0]
                chunk_arr = self.chunk_id_encoder.array

                first_sample = 0 if row == 0 else chunk_arr[row - 1][1] + 1
                last_sample = self.chunk_id_encoder.array[row][1]
                num_samples = last_sample - first_sample + 1

                full_shape = (num_samples,) + tuple(self.tensor_meta.max_shape)
                dtype = self.tensor_meta.dtype

                data_bytes = bytearray(chunk.data_bytes)
                self.cached_data = np.frombuffer(data_bytes, dtype).reshape(full_shape)
                self.cache_range = range(first_sample, last_sample + 1)

            sample = self.cached_data[global_sample_index - self.cache_range.start]  # type: ignore

            # need to copy if aslist otherwise user might modify the returned data
            # if not aslist, we already do np.array(samples) while formatting which copies
            sample = sample.copy() if aslist else sample
            sample = sample[tuple(entry.value for entry in index.values[1:])]
            samples.append(sample)
        return samples

    def get_chunks_for_sample(
        self,
        global_sample_index: int,
        copy: bool = False,
    ) -> List[BaseChunk]:
        """Retrives the `Chunk` object corresponding to `global_sample_index`.
        Args:
            global_sample_index (int): Index relative to the entire tensor representing the sample.
            copy (bool): If True and the chunk exists in a different commit to the current commit, it will be copied. Defaults to False.
        Returns:
            List[BaseChunk]: BaseChunk objects that contains `global_sample_index`.
        """
        return [
            self.get_chunk_from_chunk_id(idx, copy)
            for idx in self.chunk_id_encoder[global_sample_index]
        ]

    def validate_num_samples_is_synchronized(self):
        """Check if tensor meta length and chunk ID encoder are representing the same number of samples.
        Helpful for determining if a user has tampered with the tensor meta or the chunk ID encoder, or if
        the tensor was corruptd.

        Raises:
            CorruptedMetaError: tensor_meta and chunk_id_encoder must have the same num samples.
        """

        tensor_meta_length = self.tensor_meta.length

        # compare chunk ID encoder and tensor meta

        # update this if we change self.num_samples implementation later to use tensor meta length instead of chunk_id_encoder
        chunk_id_num_samples = self.num_samples

        if tensor_meta_length != chunk_id_num_samples:
            commit_id = self.commit_id
            tkey = get_tensor_meta_key(self.key, commit_id)
            ikey = get_chunk_id_encoder_key(self.key, commit_id)
            raise CorruptedMetaError(
                f"'{tkey}' and '{ikey}' have a record of different numbers of samples. Got {tensor_meta_length} and {chunk_id_num_samples} respectively."
            )

    def list_all_chunks(self) -> List[str]:
        """Return list of all chunks for current `version_state['commit_id']` and tensor"""
        commit_id = self.commit_id
        if commit_id == FIRST_COMMIT_ID:
            return [ChunkIdEncoder.name_from_id(chunk_id) for chunk_id in self.chunk_id_encoder.array[:, CHUNK_ID_COLUMN]]  # type: ignore
        else:
            return list(self.commit_chunk_set.chunks)  # type: ignore

    def list_all_chunks_path(self) -> List[str]:
        """Return list of paths to all chunks"""
        commit_id = self.commit_id
        return [
            get_chunk_key(self.key, chunk, commit_id)
            for chunk in self.list_all_chunks()
        ]

    def list_orphaned_chunks(self, storage):
        """Return paths for orphaned chunks (chunks what are not linked to the `current_version`)"""

        commit_id = self.commit_id
        prefix: str = f"{self.key}/chunks/"

        if commit_id != FIRST_COMMIT_ID:
            prefix = f"versions/{commit_id}/{prefix}"

        all_chunks = [
            item.replace(prefix, "") for item in storage if item.startswith(prefix)
        ]
        linked_chunks = self.list_all_chunks()

        return [
            f"{prefix}{chunk}" for chunk in all_chunks if chunk not in linked_chunks
        ]

    def clear_unusd_chunks(self, storage: StorageProvider):
        # storage.delete_multiple(self.list_orphaned_chunks(storage))
        raise NotImplementedError(
            "requires StorageProvider to be able to list all chunks"
        )

    def _pop(self):
        if self.num_samples == 0:
            raise IndexError("pop from empty tensor")
        self.commit_diff._pop()  # This will fail if the last sample was added in a previous commit
        (self._pop_sequence if self.is_sequence else self.__pop)()

    def _pop_sequence(self):
        for _ in range(*self.sequence_encoder[-1]):
            self.__pop()
        self.sequence_encoder._pop()

    def __pop(self):
        """Used only for Dataset.append"""
        self._write_initialization()
        chunk_ids, delete = self.chunk_id_encoder._pop()
        if len(chunk_ids) > 1:  # Tiled sample, delete all chunks
            del self.tile_encoder[self.num_samples - 1]
        elif not delete:  # There are other samples in the last chunk
            chunk_to_update = self.get_chunk(self.get_chunk_key_for_id(chunk_ids[0]))
            chunk_to_update._pop_sample()
        if delete:
            for chunk_key in map(self.get_chunk_key_for_id, chunk_ids):
                if (
                    self.active_appended_chunk is not None
                    and self.active_appended_chunk.key == chunk_key
                ):
                    self.active_appended_chunk = None
                    try:
                        del self.cache[chunk_key]
                    except KeyError:
                        pass
                else:
                    del self.cache[chunk_key]
        self.tensor_meta._pop()

    def write_chunk_to_storage(self, chunk):
        if chunk is None or not chunk.is_dirty:
            return
        storage = self.cache
        key = chunk.key
        storage[key] = chunk
        chunk.is_dirty = False

    @property
    def is_sequence(self):
        return self.tensor_meta.is_sequence

    @property
    def sequence_encoder_exists(self) -> bool:
        commit_id = self.commit_id
        if (
            self._sequence_encoder is not None
            and self._sequence_encoder_commit_id == commit_id
        ):
            return True
        try:
            key = get_sequence_encoder_key(self.key, commit_id)
            self.meta_cache[key]
            return True
        except KeyError:
            return False

    @property
    def _sequence_length(self):
        return self.sequence_encoder.num_samples

    @property
    def sequence_encoder(self) -> SequenceEncoder:
        """Gets the shape encoder from cache, if one is not found it creates a blank encoder.

        Raises:
            CorruptedMetaError: If shape encoding was corrupted.

        Returns:
            A SequenceEncoder instance storing the start and end indices of each sequence in the tensor.
        """

        if not self.is_sequence:
            return  # type: ignore
        commit_id = self.commit_id
        if (
            self._sequence_encoder is None
            or self._sequence_encoder_commit_id != commit_id
        ):
            commit_id = self.commit_id
            key = get_sequence_encoder_key(self.key, commit_id)
            if not self.sequence_encoder_exists:
                enc = SequenceEncoder()
                try:
                    self.meta_cache[key] = enc
                except ReadOnlyModeError:
                    pass
            else:
                enc = self.meta_cache.get_hub_object(key, SequenceEncoder)
            self._sequence_encoder = enc
            self._sequence_encoder_commit_id = commit_id
            self.meta_cache.register_hub_object(key, enc)
        return self._sequence_encoder

    def _sequence_numpy(
        self, index: Index, aslist: bool = False, use_data_cache: bool = True
    ):
        arr = self._numpy(
            self._get_flat_index_from_sequence_index(index),
            aslist=aslist,
            use_data_cache=use_data_cache,
        )
        if index.subscriptable_at(0) and index.subscriptable_at(1):
            if aslist:
                _item_length = self._sequence_item_length
                ret = []
                for i in index.values[0].indices(self._sequence_length):
                    item_length = _item_length or index.length_at(
                        1, -int(np.subtract(*self.sequence_encoder[i]))
                    )
                    ret.append(arr[:item_length])
                    arr = arr[item_length:]
                return ret
            else:
                try:
                    return arr.reshape(  # type: ignore
                        index.length_at(0, self._sequence_length), -1, *arr.shape[1:]  # type: ignore
                    )
                except ValueError as ve:
                    raise DynamicTensorNumpyError(self.key, index, "shape") from ve
        return arr

    def _translate_2d_index(
        self, x: Optional[IndexEntry] = None, y: Optional[IndexEntry] = None
    ) -> IndexEntry:
        x = x or IndexEntry()
        y = y or IndexEntry()
        _item_length = self._sequence_item_length
        if _item_length is None:

            def idx0_gen():
                for i in x.indices(self._sequence_length):
                    s, e = self.sequence_encoder[i]
                    for j in y.indices(e - s):
                        yield s + j

        else:

            def idx0_gen():
                for i in x.indices(self._sequence_length):
                    for j in y.indices(_item_length):
                        yield i * _item_length + j

        idx0_gen.__len__ = (  # type: ignore
            (
                lambda: sum(
                    [
                        y.length(-np.subtract(*self.sequence_encoder[i]))
                        for i in x.indices(self._sequence_length)
                    ]
                )
            )
            if _item_length is None
            else (lambda: x.length(self._sequence_length) * y.length(_item_length))  # type: ignore
        )
        return IndexEntry(idx0_gen)  # type: ignore

    def _get_flat_index_from_sequence_index(self, index: Index) -> Index:
        if len(index) == 1:
            index = Index([index.values[0], IndexEntry()])
        if index.values[0].is_trivial() and index.values[1].is_trivial():
            return Index([IndexEntry(), *index.values[2:]])
        if index.subscriptable_at(0) or index.subscriptable_at(1):
            idx0 = self._translate_2d_index(index.values[0], index.values[1])
            return Index([idx0, *index.values[2:]])  # type: ignore
        return Index(
            [
                IndexEntry(
                    self.sequence_encoder[index.values[0].value][0]  # type: ignore
                    + index.values[1].value
                ),
                *index.values[2:],
            ]
        )

    def _get_flat_samples_for_sequence_update(self, samples, index: Index):
        ndim = self.ndim(index)
        if isinstance(samples, np.ndarray):
            if index.subscriptable_at(0) and index.subscriptable_at(1):
                diff = ndim - samples.ndim
                if diff < 0:
                    samples, diff = samples.reshape(samples.shape[-ndim:]), 0
                if diff > 1:
                    return samples.reshape(1, *samples.shape).repeat(
                        self._translate_2d_index(*index.values[:2]).length(None), 0  # type: ignore
                    )
                elif diff == 1:
                    return (
                        samples.reshape(1, *samples.shape)
                        .repeat(index.length_at(0, self._sequence_length), 0)
                        .reshape(-1, *samples.shape[1:])
                    )
                else:
                    return samples.reshape(-1, *samples.shape[2:])
            return samples
        elif isinstance(samples, (str, bytes)):  # treated as scalars
            return samples
        elif isinstance(samples, Iterable):
            # Note: broadcasting is not supported here
            if index.subscriptable_at(0) and index.subscriptable_at(1):
                return list(chain(*samples))
            return samples
        else:
            return samples  # scalars

    def _sequence_update(
        self,
        index: Index,
        samples: Union[np.ndarray, Sequence[InputSample], InputSample],
        operator: Optional[str] = None,
        link_callback: Optional[Callable] = None,
    ):
        flat_idx = self._get_flat_index_from_sequence_index(index)
        flat_samples = self._get_flat_samples_for_sequence_update(samples, index)
        flat_verified_samples: List = self._update(
            flat_idx,
            flat_samples,
            operator,
            update_commit_diff=False,
            link_callback=link_callback,
        )
        i = 0
        verified_samples: Optional[List] = None
        if flat_verified_samples:
            verified_samples = []
            for sample in samples:  # type: ignore
                verified_sample = []
                for _ in sample:  # type: ignore
                    verified_sample.append(flat_verified_samples[i])
                    i += 1
                verified_samples.append(verified_sample)

        list(
            map(
                self.commit_diff.update_data,
                index.values[0].indices(self._sequence_length),
            )
        )
        if link_callback:
            ls = verified_samples or samples

            if isinstance(ls, np.ndarray):
                broadcast = ls.ndim < self.ndim(index)
            elif isinstance(ls, (bytes, str)):  # sacalars:
                broadcast = True
            elif isinstance(ls, Iterable):
                broadcast = False
            else:
                broadcast = True
            seq_len = self._sequence_length
            if broadcast:
                ls = repeat(ls)  # type: ignore
            for i, sample in zip(index.values[0].indices(seq_len), ls):  # type: ignore
                link_callback(
                    i, sub_index=Index(index.values[1:]), new_sample=sample, flat=False
                )

    @property
    def _sequence_item_length(self):
        enc = self.sequence_encoder
        nrows = len(enc._encoded)
        if nrows == 0:
            return 0
        if nrows == 1:
            s, e = enc[0]
            return e - s
        else:
            return None

    def shape(
        self, index: Index, sample_shape_provider: Optional[Callable] = None
    ) -> Tuple[Optional[int], ...]:
        shape = self.shape_interval.astuple()
        idxs = index.values
        skip_dims = 0
        if self.is_sequence:
            if not idxs[0].subscriptable():
                shape = shape[1:]
                skip_dims += 1
            if len(idxs) > 1 and not idxs[1].subscriptable():
                shape = shape[1:]
                skip_dims += 1
        else:
            if None in shape or self.tensor_meta.is_link:
                if not idxs[0].subscriptable():
                    if self.tensor_meta.htype in ("text", "json"):
                        shape = (1,)
                    else:
                        if sample_shape_provider:
                            try:
                                shape = sample_shape_provider(idxs[0].value)  # type: ignore
                            except IndexError:  # Happens during transforms, sample shape tensor is not populated yet
                                shape = self.read_shape_for_sample(idxs[0].value)  # type: ignore
                        else:
                            shape = self.read_shape_for_sample(idxs[0].value)  # type: ignore
                    skip_dims += 1
            elif not idxs[0].subscriptable():
                shape = shape[1:]
                skip_dims += 1
        shape = list(shape)  # type: ignore
        squeeze_dims = set()
        for i, idx in enumerate(idxs[skip_dims:]):
            if idx.subscriptable():
                shape[i] = idx.length(shape[i])  # type: ignore
            else:
                squeeze_dims.add(i)
        return tuple(shape[i] for i in range(len(shape)) if i not in squeeze_dims)

    def ndim(self, index: Optional[Index] = None) -> int:
        ndim = len(self.tensor_meta.min_shape) + 1
        if self.is_sequence:
            ndim += 1
        if index:
            for idx in index.values:
                if not idx.subscriptable():
                    ndim -= 1
        return ndim

    @property
    def shape_interval(self) -> ShapeInterval:
        """Returns a `ShapeInterval` object that describes this tensor's shape more accurately. Length is included.

        Note:
            If you are expecting a `tuple`, use `tensor.shape` instead.

        Example:
            >>> tensor.append(np.zeros((10, 10)))
            >>> tensor.append(np.zeros((10, 15)))
            >>> tensor.shape_interval
            ShapeInterval(lower=(2, 10, 10), upper=(2, 10, 15))
            >>> str(tensor.shape_interval)
            (2, 10, 10:15)

        Returns:
            ShapeInterval: Object containing `lower` and `upper` properties.
        """
        meta = self.tensor_meta
        if self.is_sequence:
            length = [
                self._sequence_length,
                self._sequence_item_length,
            ]
        else:
            length = [meta.length]
        min_shape = length + list(meta.min_shape)
        max_shape = length + list(meta.max_shape)

        return ShapeInterval(min_shape, max_shape)

    def _transform_callback(self, sample, flat: Optional[bool]):
        """Used in transforms to handle linked tensors."""
        assert self._all_chunk_engines is not None
        for k, v in self.tensor_meta.links.items():
            if flat is None or v["flatten_sequence"] == flat:
                self._all_chunk_engines[k].extend(
                    [get_link_transform(v["append"])(sample)]
                )
