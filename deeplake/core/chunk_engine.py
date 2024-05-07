from collections import OrderedDict
from deeplake.client.log import logger
import deeplake
import numpy as np
from tqdm import tqdm
from typing import (
    Any,
    Callable,
    Dict,
    Optional,
    Sequence,
    Union,
    List,
    Tuple,
)
from deeplake.api.info import Info
from deeplake.core.link_creds import LinkCreds
from deeplake.core.linked_sample import LinkedSample, read_linked_sample
from deeplake.core.meta.encode.base_encoder import LAST_SEEN_INDEX_COLUMN
from deeplake.core.serialize import HEADER_SIZE_BYTES, text_to_bytes
from deeplake.core.tensor_link import (
    cast_to_type,
    extend_downsample,
    get_link_transform,
)
from deeplake.core.linked_tiled_sample import LinkedTiledSample
from deeplake.core.version_control.commit_diff import CommitDiff
from deeplake.core.partial_reader import PartialReader
from deeplake.core.version_control.commit_node import CommitNode  # type: ignore
from deeplake.core.version_control.commit_chunk_map import CommitChunkMap  # type: ignore
from typing import Any, Dict, List, Optional, Sequence, Union, Callable
from deeplake.core.meta.encode.tile import TileEncoder
from deeplake.core.storage.provider import StorageProvider
from deeplake.core.storage import S3Provider, GCSProvider, AzureProvider, MemoryProvider
from deeplake.core.tiling.deserialize import (
    combine_chunks,
    translate_slices,
    coalesce_tiles,
)
from deeplake.core.tiling.serialize import break_into_tiles
from deeplake.core.polygon import Polygons
from deeplake.util.casting import get_empty_text_like_sample, intelligent_cast
from deeplake.util.empty_sample import is_empty_list
from deeplake.util.shape_interval import ShapeInterval
from deeplake.constants import (
    DEFAULT_MAX_CHUNK_SIZE,
    FIRST_COMMIT_ID,
    PARTIAL_NUM_SAMPLES,
    FAST_EXTEND_BAIL,
    RANDOM_MAX_ALLOWED_CHUNK_SIZE,
    RANDOM_MINIMAL_CHUNK_SIZE,
    DEFAULT_MAX_CHUNK_SIZE,
    FIRST_COMMIT_ID,
    PARTIAL_NUM_SAMPLES,
    DEFAULT_TILING_THRESHOLD,
)
from deeplake.core.chunk.base_chunk import BaseChunk, InputSample
from deeplake.core.chunk.chunk_compressed_chunk import ChunkCompressedChunk
from deeplake.core.chunk.sample_compressed_chunk import SampleCompressedChunk
from deeplake.core.chunk.uncompressed_chunk import UncompressedChunk
from deeplake.core.fast_forwarding import ffw_chunk_id_encoder
from deeplake.core.index.index import Index, IndexEntry
from deeplake.core.meta.encode.chunk_id import CHUNK_ID_COLUMN, ChunkIdEncoder
from deeplake.core.meta.encode.sequence import SequenceEncoder
from deeplake.core.meta.encode.pad import PadEncoder
from deeplake.core.meta.tensor_meta import (
    TensorMeta,
    _validate_required_htype_overwrites,
)
from deeplake.core.storage.lru_cache import LRUCache
from deeplake.util.casting import get_dtype, get_htype
from deeplake.core.sample import Sample
from deeplake.util.chunk_engine import (
    check_samples_type,
    make_sequence,
    check_suboptimal_chunks,
    check_sample_shape,
)
from deeplake.util.keys import (
    get_chunk_id_encoder_key,
    get_sequence_encoder_key,
    get_pad_encoder_key,
    get_tensor_commit_diff_key,
    get_tensor_meta_key,
    get_chunk_key,
    get_tensor_commit_chunk_map_key,
    get_tensor_commit_chunk_map_key,
    get_tensor_meta_key,
    get_tensor_tile_encoder_key,
    get_tensor_info_key,
)
from deeplake.util.exceptions import (
    GetChunkError,
    CorruptedMetaError,
    DynamicTensorNumpyError,
    GetDataFromLinkError,
    ReadOnlyModeError,
    ReadSampleFromChunkError,
    SampleAppendError,
    SampleHtypeMismatchError,
    SampleUpdateError,
)
from deeplake.util.remove_cache import get_base_storage
from deeplake.util.image import convert_sample, convert_img_arr
from deeplake.util.class_label import convert_to_idx, convert_to_hash
from deeplake.compression import (
    BYTE_COMPRESSION,
    VIDEO_COMPRESSIONS,
    get_compression_type,
)
from deeplake.core.sample import Sample
from itertools import chain, repeat
from collections.abc import Iterable
from PIL import Image  # type: ignore
from functools import partial
from concurrent import futures
from concurrent.futures import ThreadPoolExecutor
from deeplake.core.storage.lru_cache import _get_nbytes
import threading
import time


class ChunkEngine:
    def __init__(
        self,
        key: str,
        cache: LRUCache,
        version_state: Dict[str, Any],
        meta_cache: Optional[LRUCache] = None,
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
        self.name = version_state["tensor_names"].get(self.key)
        self.compression = None
        self.chunk_class = BaseChunk

        self._tensor_meta: Optional[TensorMeta] = None
        self._tensor_meta_commit_id: Optional[str] = None

        self._chunk_id_encoder: Optional[ChunkIdEncoder] = None
        self._chunk_id_encoder_commit_id: Optional[str] = None

        self._sequence_encoder: Optional[SequenceEncoder] = None
        self._sequence_encoder_commit_id: Optional[str] = None

        self._pad_encoder: Optional[PadEncoder] = None
        self._pad_encoder_commit_id: Optional[str] = None

        self._tile_encoder: Optional[TileEncoder] = None
        self._tile_encoder_commit_id: Optional[str] = None

        self._commit_chunk_map: Optional[CommitChunkMap] = None
        self._commit_chunk_map_commit_id: Optional[str] = None

        self._commit_diff: Optional[CommitDiff] = None
        self._commit_diff_commit_id: Optional[str] = None

        self._active_appended_chunk: Optional[BaseChunk] = None
        self._active_updated_chunk: Optional[BaseChunk] = None

        self._info: Optional[Info] = None
        self._info_commit_id: Optional[str] = None

        self._all_chunk_engines: Optional[Dict[str, ChunkEngine]] = None
        self._is_temp_label_tensor: bool = False
        self._hash_label_map: Dict[int, str] = OrderedDict()
        self._sample_compression = None
        self._chunk_compression = None

        tensor_meta = self.tensor_meta
        self.name = tensor_meta.name or self.key
        numpy_extend_optimization_enabled = False

        if tensor_meta.sample_compression:
            self._sample_compression = self.compression = tensor_meta.sample_compression
            self.chunk_class = SampleCompressedChunk

        elif tensor_meta.chunk_compression:
            self._chunk_compression = self.compression = tensor_meta.chunk_compression
            self.chunk_class = ChunkCompressedChunk
            if get_compression_type(tensor_meta.chunk_compression) == BYTE_COMPRESSION:
                numpy_extend_optimization_enabled = True
        else:
            self.chunk_class = UncompressedChunk
            numpy_extend_optimization_enabled = True

        self._numpy_extend_optimization_enabled = numpy_extend_optimization_enabled

        self.cache_enabled = True
        self.cached_data: Optional[np.ndarray] = None
        self.cache_range: range = range(0)

        self._chunk_args = None
        self._num_samples_per_chunk: Optional[int] = None
        self.write_initialization_done = False
        self.start_chunk = None
        self.link_creds: Optional[LinkCreds] = None

    @property
    def sample_compression(self):
        return self._sample_compression

    @property
    def chunk_compression(self):
        return self._chunk_compression

    @property
    def is_data_cachable(self):
        if self.cache_enabled:
            tensor_meta = self.tensor_meta
            return (
                self.chunk_class == UncompressedChunk
                and tensor_meta.htype not in ["text", "json", "list", "polygon", "tag"]
                and tensor_meta.max_shape
                and (tensor_meta.max_shape == tensor_meta.min_shape)
                and (np.prod(tensor_meta.max_shape) < 20)
            )
        return False

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
    def tiling_threshold(self):
        return (
            getattr(self.tensor_meta, "tiling_threshold", None)
            or DEFAULT_TILING_THRESHOLD
            or self.min_chunk_size
        )

    @property
    def chunk_args(self):
        if self._chunk_args is None:
            self._chunk_args = [
                self.min_chunk_size,
                self.max_chunk_size,
                self.tiling_threshold,
                self.tensor_meta,
                self.compression,
            ]
        return self._chunk_args

    @property
    def min_chunk_size(self):
        # only the last chunk may be less than this
        return self.max_chunk_size // 2

    @property
    def tensor_meta(self):
        commit_id = self.commit_id
        if self._tensor_meta is None or self._tensor_meta_commit_id != commit_id:
            key = get_tensor_meta_key(self.key, commit_id)
            self._tensor_meta = self.meta_cache.get_deeplake_object(key, TensorMeta)
            self._tensor_meta_commit_id = commit_id
            self.meta_cache.register_deeplake_object(key, self._tensor_meta)
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
                enc = ChunkIdEncoder(dtype=np.uint64)
                try:
                    self.meta_cache[key] = enc
                except ReadOnlyModeError:
                    pass
            else:
                enc = self.meta_cache.get_deeplake_object(key, ChunkIdEncoder)
            self._chunk_id_encoder = enc
            self._chunk_id_encoder_commit_id = commit_id
            self.meta_cache.register_deeplake_object(key, enc)
        return self._chunk_id_encoder

    @property
    def commit_chunk_map(self) -> Optional[CommitChunkMap]:
        """Gets the commit chunk map from cache, if one is not found it creates a blank one.

        Returns:
            Optional[CommitChunkMap]: The commit chunk map keeps track of all the chunks present in the current commit, returns None for the first commit.
        """
        commit_id = self.commit_id
        if commit_id == FIRST_COMMIT_ID:
            # the first commit doesn't need a commit chunk map
            return None
        if (
            self._commit_chunk_map is None
            or self._commit_chunk_map_commit_id != commit_id
        ):
            key = get_tensor_commit_chunk_map_key(self.key, commit_id)
            if not self.commit_chunk_map_exists:
                cmap = CommitChunkMap()
                try:
                    self.meta_cache[key] = cmap
                except ReadOnlyModeError:
                    pass
            else:
                cmap = self.meta_cache.get_deeplake_object(key, CommitChunkMap)
            self._commit_chunk_map = cmap
            self._commit_chunk_map_commit_id = commit_id
            self.meta_cache.register_deeplake_object(key, cmap)
        return self._commit_chunk_map

    @property
    def commit_chunk_map_exists(self) -> bool:
        """Checks if the commit chunk map exists for the given tensor in the current commit."""
        commit_id = self.commit_id
        if (
            self._commit_chunk_map is not None
            and self._commit_chunk_map_commit_id == commit_id
        ):
            return True

        try:
            key = get_tensor_commit_chunk_map_key(self.key, commit_id)
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
                diff = self.meta_cache.get_deeplake_object(key, CommitDiff)
            self._commit_diff = diff
            self._commit_diff_commit_id = commit_id
            self.meta_cache.register_deeplake_object(key, diff)
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
                enc = self.meta_cache.get_deeplake_object(key, TileEncoder)
            self._tile_encoder = enc
            self._tile_encoder_commit_id = commit_id
            self.meta_cache.register_deeplake_object(key, enc)
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
        """Total length of tensor (includes samples in sequences)
        Ignores any applied indexing and returns the total length.
        """
        return self.tensor_meta.length

    @property
    def tensor_length(self) -> int:
        """Length of primary axis of tensor (does not include samples in sequences)"""
        return self._sequence_length or self.tensor_meta.length

    @property
    def last_chunk_key(self) -> str:
        last_chunk_name = self.last_appended_chunk_name
        commit_id, tkey = self.get_chunk_commit(last_chunk_name)
        return get_chunk_key(tkey, last_chunk_name, commit_id)

    def get_chunk_key_for_id(self, chunk_id) -> str:
        chunk_name = ChunkIdEncoder.name_from_id(chunk_id)
        commit_id, tkey = self.get_chunk_commit(chunk_name)
        return get_chunk_key(tkey, chunk_name, commit_id)

    @property
    def active_appended_chunk(self):
        return self._active_appended_chunk

    @active_appended_chunk.setter
    def active_appended_chunk(self, value):
        if self.active_appended_chunk is not None:
            self.cache.remove_deeplake_object(self.active_appended_chunk.key)
        self._active_appended_chunk = value
        if value is not None:
            self.cache.register_deeplake_object(value.key, value)

    @property
    def active_updated_chunk(self):
        return self._active_updated_chunk

    @active_updated_chunk.setter
    def active_updated_chunk(self, value):
        if self.active_updated_chunk is not None:
            self.cache.remove_deeplake_object(self.active_updated_chunk.key)
        self._active_updated_chunk = value
        if value is not None:
            self.cache.register_deeplake_object(value.key, value)

    @property
    def last_appended_chunk_name(self) -> str:
        return self.chunk_id_encoder.get_name_for_chunk(-1)

    @property
    def last_appended_chunk_id(self) -> str:
        return self.chunk_id_encoder.get_id_for_chunk(-1)

    def last_appended_chunk(self, allow_copy=True) -> Optional[BaseChunk]:
        last_index = self.num_samples - 1
        if self.num_chunks == 0 or last_index in self.tile_encoder:
            return None
        chunk_name = self.last_appended_chunk_name
        chunk_commit_id, tkey = self.get_chunk_commit(chunk_name)
        chunk_key = get_chunk_key(tkey, chunk_name, chunk_commit_id)
        chunk = self.get_chunk(chunk_key)
        chunk.key = chunk_key
        chunk.id = self.last_appended_chunk_id
        if chunk_commit_id != self.commit_id:
            if not allow_copy:
                return None
            chunk = self.copy_chunk_to_new_commit(chunk, chunk_name)
        if (
            self.active_appended_chunk is not None
            and self.active_appended_chunk.key != chunk_key
        ):
            self.write_chunk_to_storage(self.active_appended_chunk)
        self.active_appended_chunk = chunk
        return chunk

    def get_chunk(self, chunk_key: str, partial_chunk_bytes=0) -> BaseChunk:
        chunk = self.cache.get_deeplake_object(
            chunk_key,
            self.chunk_class,
            self.chunk_args,
            partial_bytes=partial_chunk_bytes,
        )
        if not partial_chunk_bytes and isinstance(chunk.data_bytes, PartialReader):
            chunk._make_data_bytearray()
        return chunk

    def get_chunk_from_chunk_id(
        self, chunk_id, copy: bool = False, partial_chunk_bytes=0
    ) -> BaseChunk:
        chunk_key = None
        try:
            chunk_name = ChunkIdEncoder.name_from_id(chunk_id)
            chunk_commit_id, tkey = self.get_chunk_commit(chunk_name)
            chunk_key = get_chunk_key(tkey, chunk_name, chunk_commit_id)
            chunk = self.get_chunk(chunk_key, partial_chunk_bytes=partial_chunk_bytes)
            chunk.key = chunk_key
            chunk.id = chunk_id
            if copy and chunk_commit_id != self.commit_id:
                chunk = self.copy_chunk_to_new_commit(chunk, chunk_name)
            return chunk
        except Exception as e:
            raise GetChunkError(chunk_key, cause=e) from e

    def get_video_chunk(self, chunk_id, copy: bool = False):
        """Returns video chunks. Chunk will contain presigned url to the video instead of data if the chunk is large."""
        chunk_name = ChunkIdEncoder.name_from_id(chunk_id)
        chunk_commit_id, tkey = self.get_chunk_commit(chunk_name)
        chunk_key = get_chunk_key(tkey, chunk_name, chunk_commit_id)

        base_storage = self.base_storage
        stream = False
        if isinstance(base_storage, (S3Provider, GCSProvider, AzureProvider)):
            chunk_size = base_storage.get_object_size(chunk_key)
            stream = chunk_size > self.min_chunk_size
            if stream:
                chunk = self.cache.get_deeplake_object(
                    chunk_key, self.chunk_class, meta=self.chunk_args, url=True
                )
        if not stream:
            chunk = self.cache.get_deeplake_object(
                chunk_key, self.chunk_class, meta=self.chunk_args
            )
        chunk.key = chunk_key
        chunk.id = chunk_id
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
        if self.commit_chunk_map is not None:
            self.commit_chunk_map.add(chunk_name)
        return chunk

    def get_chunk_commit(self, chunk_name) -> Tuple[str, str]:
        """Returns the commit id and tensor key that contains the chunk_name."""
        cur_node: Optional[CommitNode] = self.version_state["commit_node"]
        key = self.key
        while cur_node is not None:
            commit_id = cur_node.commit_id
            chunk_map_key = get_tensor_commit_chunk_map_key(key, commit_id)
            try:
                # the first commit doesn't contain a chunk map, don't repeatedly try to fetch from storage
                if commit_id == FIRST_COMMIT_ID:
                    chunk_map = dict()
                else:
                    chunk_map = self.meta_cache.get_deeplake_object(
                        chunk_map_key, CommitChunkMap
                    ).chunks
            except Exception:
                commit_chunk_map = CommitChunkMap()
                try:
                    self.meta_cache[chunk_map_key] = commit_chunk_map
                except ReadOnlyModeError:
                    # put CommitChunkMap in deeplake_objects to keep in cache temporarily, but won't write to storage
                    # this shouldn't happen in latest version of deeplake, chunk map would always be present
                    self.meta_cache.deeplake_objects[chunk_map_key] = commit_chunk_map
                chunk_map = dict()
            v = chunk_map.get(chunk_name)
            if v is not None:
                commit_id = v.get("commit_id", commit_id)
                key = v.get("key", key)
                return commit_id, key
            cur_node = cur_node.parent  # type: ignore
        # the first commit doesn't have a commit chunk map, so any chunk that wasn't found belongs to the first commit
        return FIRST_COMMIT_ID, key

    def _write_initialization(self):
        ffw_chunk_id_encoder(self.chunk_id_encoder)

    def _convert_to_list(self, samples):
        return False

    def check_each_sample(self, samples, verify=True, ignore_errors=False):
        # overridden in LinkedChunkEngine
        return

    def _link_tensor_to_samples(self, samples):
        for i, sample in enumerate(samples):
            if (
                isinstance(sample, deeplake.core.tensor.Tensor)
                and sample.is_link
                and not (
                    sample.index.values[0].subscriptable()
                    or len(sample.index.values) > 1
                )
            ):
                sample = sample._linked_sample()
                samples[i] = sample

    def _sanitize_samples(self, samples, ignore_errors=False):
        check_samples_type(samples)
        samples = self._prepare_samples_for_link_callback(samples)
        tensor_meta = self.tensor_meta
        all_empty = all(sample is None for sample in samples)
        if tensor_meta.htype is None and not all_empty:
            htype = get_htype(samples)
            if tensor_meta.dtype is not None:
                _validate_required_htype_overwrites(
                    htype,
                    {
                        "sample_compression": tensor_meta.sample_compression,
                        "chunk_compression": tensor_meta.chunk_compression,
                        "dtype": tensor_meta.dtype,
                    },
                )
            tensor_meta.set_htype(htype)
        if tensor_meta.dtype is None and not all_empty:
            if tensor_meta.is_link:
                try:
                    # download one sample to get dtype
                    sample = next(filter(lambda x: x is not None, samples))
                    assert isinstance(
                        sample, LinkedSample
                    ), "Sample must be LinkedSample"
                    dtype = np.dtype(
                        read_linked_sample(
                            sample.path, sample.creds_key, self.link_creds, True
                        )._typestr
                    )
                except:
                    # assume uint8 if download fails
                    dtype = np.dtype("uint8")
            else:
                non_empty_samples = list(filter(lambda x: x is not None, samples))
                for sample in non_empty_samples:
                    try:
                        dtype = get_dtype(sample)
                        break
                    except:
                        pass
                else:
                    if not ignore_errors:
                        raise ValueError("Could not determine dtype of samples")
            tensor_meta.set_dtype(dtype)
        if self._convert_to_list(samples):
            samples = list(samples)
        if self._is_temp_label_tensor:
            samples = convert_to_hash(samples, self._hash_label_map)
        elif tensor_meta.htype in ("image.gray", "image.rgb"):
            mode = "L" if tensor_meta.htype == "image.gray" else "RGB"
            converted = []
            for sample in samples:
                try:
                    if isinstance(sample, Sample):
                        converted.append(convert_sample(sample, mode))
                    elif isinstance(sample, np.ndarray):
                        converted.append(convert_img_arr(sample, mode))
                    else:
                        raise SampleHtypeMismatchError(tensor_meta.htype, type(sample))
                except Exception:
                    if ignore_errors:
                        continue
                    raise
            samples = converted
        elif tensor_meta.htype == "class_label":
            samples = self._convert_class_labels(samples)
        elif tensor_meta.htype == "polygon":
            samples = [
                p if isinstance(p, Polygons) else Polygons(p, dtype=tensor_meta.dtype)
                for p in samples
            ]
        elif tensor_meta.htype == "tag":
            samples = [
                sample if isinstance(sample, list) else [sample] for sample in samples
            ]
        return samples

    def _convert_class_labels(self, samples):
        tensor_info_path = get_tensor_info_key(self.key, self.commit_id)
        try:
            tensor_info = self.cache.get_deeplake_object(tensor_info_path, Info)
        except KeyError:
            tensor_info = Info()
        self.cache.register_deeplake_object(tensor_info_path, tensor_info)
        tensor_name = self.tensor_meta.name or self.key
        class_names = tensor_info.class_names
        labels, additions = convert_to_idx(samples, class_names)
        if additions:
            for new in additions:
                class_names.append(new[0])
                logger.info(
                    f"'{new[0]}' added to {tensor_name}.info.class_names at index {new[1]}"
                )
            tensor_info.class_names = class_names
            tensor_info.is_dirty = True
        self.commit_diff.modify_info()
        self.cache.maybe_flush()
        return labels

    def _samples_to_chunks(
        self,
        samples,
        start_chunk: Optional[BaseChunk] = None,
        register: bool = True,
        update_commit_diff: bool = False,
        update_tensor_meta: bool = True,
        start_chunk_row: Optional[int] = None,
        progressbar: bool = False,
        register_creds: bool = True,
        pg_callback=None,
        return_samples: bool = False,
        ignore_errors: bool = False,
    ):
        """Add samples to chunks, in case if there is a space on the start_chunk,
        othewise creating new chunk and append samples to newly created chunk

        Args:
            samples (List[Any]): Paramter that shows the list of samples to be added to the chunk
            start_chunk (BaseChunk, Optional): Parameter that points to the chunk on which the samples should be added
            register (bool): Parameter that shows if we need to register the chunk
            update_commit_diff (bool): Parameter that shows if we need to update the commit diffs
            update_tensor_meta (bool): Parameter that shows if it is needed to update tensor metas, this will be false in case of rechunking at the meta will not be changed
            start_chunk_row (int, Optional): Parameter that shows the chunk row that needs to be updated, those params are needed only in rechunking phase.
            progressbar (bool): Parameter that shows if need to show sample insertion progress
            register_creds (bool): Parameter that shows if need to register the creds_key of the sample
            pg_callback: Progress bar callback parameter
            return_samples (bool): Returns successfully added samples if ``True``.
            ignore_errors (bool): Skips samples that cause errors, if possible.

        Returns:
            Tuple[List[BaseChunk], Dict[Any, Any]]
        """
        extending = start_chunk_row is None and register
        lengths = None
        orig_meta_length = self.tensor_meta.length
        incoming_num_samples = len(samples)
        enc_ids: List[Optional[str]] = []
        enc_count = [0]
        if extending:
            if self.tensor_meta.htype == "text" and (
                self.chunk_class != SampleCompressedChunk
            ):
                lengths = np.zeros(len(samples), dtype=np.uint32)
                for i, s in enumerate(samples):
                    try:
                        s = s.numpy()
                    except AttributeError:
                        pass
                    try:
                        if s.dtype.name[:3] == "str":
                            lengths[i] = len(str(s.reshape(())))
                    except AttributeError:
                        try:
                            lengths[i] = s.__len__()
                        except AttributeError:  # None
                            lengths[i] = 0
                        except TypeError:  # Numpy scalar str
                            lengths[i] = str(s).__len__()
        extra_args = {"lengths": lengths}
        current_chunk = start_chunk
        updated_chunks: List[Optional[str]] = []
        if current_chunk is None:
            current_chunk = self._create_new_chunk(
                register and start_chunk_row is not None
            )
            current_chunk._update_tensor_meta_length = False
            if not register:
                updated_chunks.append(current_chunk.id)
            if extending:
                enc_ids.append(current_chunk.id)
        else:
            current_chunk._update_tensor_meta_length = False
            if extending:
                enc_ids.append(None)
        enc = self.chunk_id_encoder
        tiles: Dict[int, Tuple[Tuple[int, ...], Tuple[int, ...]]] = {}
        if register and update_commit_diff:
            commit_diff = self.commit_diff
        if progressbar:
            pbar = tqdm(total=len(samples))
        if not isinstance(samples, list) and not (
            isinstance(samples, np.ndarray) and self._numpy_extend_optimization_enabled
        ):
            # Note: in the future we can get rid of this conversion of sample compressed chunks too by predicting the compression ratio.
            samples = list(samples)
        verified_samples = []
        current_chunk_full = False
        while len(samples) > 0:
            if current_chunk_full:
                num_samples_added = 0
                current_chunk_full = False
            else:
                initial_num_samples = len(samples)
                num_samples_added = current_chunk.extend_if_has_space(
                    samples, update_tensor_meta=update_tensor_meta, ignore_errors=ignore_errors, **extra_args  # type: ignore
                )  # type: ignore
                skipped_num_samples = initial_num_samples - len(samples)
                incoming_num_samples -= skipped_num_samples
                if register_creds:
                    self.register_new_creds(num_samples_added, samples)
            if num_samples_added == 0:
                current_chunk = self._create_new_chunk(
                    register and start_chunk_row is not None, row=start_chunk_row
                )
                current_chunk._update_tensor_meta_length = False
                if start_chunk_row is not None:
                    start_chunk_row += 1
                elif register:
                    enc_ids.append(current_chunk.id)
                    enc_count.append(0)
                if not register:
                    updated_chunks.append(current_chunk.id)
            elif num_samples_added == PARTIAL_NUM_SAMPLES:
                sample = samples[0]
                if self.tensor_meta.is_link:
                    verified_samples.append(sample)
                else:
                    if sample.is_first_write:
                        verified_samples.append(sample)
                num_samples_added, samples, lengths = self._handle_tiled_sample(
                    enc,
                    register,
                    samples,
                    orig_meta_length,
                    incoming_num_samples,
                    start_chunk_row,
                    enc_count,
                    tiles,
                    lengths,
                )
                if len(samples) > 0:
                    current_chunk = self._create_new_chunk(
                        register and start_chunk_row is not None, row=start_chunk_row
                    )
                    current_chunk._update_tensor_meta_length = False
                    if start_chunk_row is not None:
                        start_chunk_row += 1
                    elif register:
                        enc_ids.append(current_chunk.id)
                        enc_count.append(0)
                    if not register:
                        updated_chunks.append(current_chunk.id)
            elif num_samples_added == FAST_EXTEND_BAIL:
                num_samples_added = 0
                samples = list(samples)
            else:
                current_chunk_full = True
                verified_samples.extend(samples[:num_samples_added])
                num_samples_added, samples, lengths = self._handle_one_or_more_samples(
                    enc,
                    register,
                    samples,
                    num_samples_added,
                    updated_chunks,
                    start_chunk_row,
                    current_chunk,
                    enc_count,
                    lengths,
                )
            if progressbar:
                pbar.update(num_samples_added)
            elif pg_callback is not None:
                pg_callback(num_samples_added)
        if extending:
            if enc_ids[0] is None:
                enc_ids.pop(0)
                start_chunk_incr = enc_count.pop(0)
                enc._encoded[-1, 1] += start_chunk_incr
                enc.is_dirty = True
            if enc_count:
                enc_arr = enc._encoded
                n = len(enc_arr)
                if n:
                    enc_count[0] += enc_arr[-1, 1]
                else:
                    enc_count[0] -= 1
                enc_last_seen = np.cumsum(enc_count, dtype=np.uint64)
                arr = np.zeros((n + len(enc_ids), 2), dtype=np.uint64)
                if n:
                    arr[:n] = enc_arr
                new = arr[n:]
                new[:, 0] = enc_ids
                new[:, 1] = enc_last_seen
                enc._encoded = arr
                enc.is_dirty = True
            self.tensor_meta.update_length(incoming_num_samples)
        if register:
            if update_commit_diff:
                commit_diff.add_data(incoming_num_samples)
            tenc = self.tile_encoder
            tenc.entries.update(tiles)
            tenc.is_dirty = True
        if progressbar:
            pbar.close()

        if return_samples:
            return verified_samples

        if not register:
            return updated_chunks, tiles

    def _handle_one_or_more_samples(
        self,
        enc: ChunkIdEncoder,
        register,
        samples,
        num_samples_added,
        updated_chunks,
        start_chunk_row,
        current_chunk,
        enc_count,
        lengths,
    ):
        if not register and not updated_chunks:
            updated_chunks.append(current_chunk)
        num_samples_added = int(num_samples_added)
        if register:
            if start_chunk_row is not None:
                enc.register_samples(num_samples_added, row=start_chunk_row)
            else:
                enc_count[-1] += num_samples_added
        if lengths is not None:
            lengths = lengths[num_samples_added:]
        samples = samples[num_samples_added:]
        return num_samples_added, samples, lengths

    def _handle_tiled_sample(
        self,
        enc: ChunkIdEncoder,
        register,
        samples,
        orig_meta_length,
        incoming_num_samples,
        start_chunk_row,
        enc_count,
        tiles,
        lengths,
    ):
        sample = samples[0]
        if sample.is_first_write:
            if register:
                if start_chunk_row is not None:
                    enc.register_samples(1)
                else:
                    enc_count[-1] += 1
        if sample.is_last_write:
            tiles[
                incoming_num_samples - len(samples) + bool(register) * orig_meta_length
            ] = (
                sample.sample_shape,
                sample.tile_shape,
            )
            samples = samples[1:]
            if lengths is not None:
                lengths = lengths[1:]
            num_samples_added = 1
        else:
            num_samples_added = 0
        return num_samples_added, samples, lengths

    def register_new_creds(self, num_samples_added, samples):
        return

    def update_creds(self, sample_index, sample):
        return

    def _extend(
        self,
        samples,
        progressbar,
        pg_callback=None,
        update_commit_diff=True,
        ignore_errors=False,
        verified_samples=None,
    ):
        if isinstance(samples, deeplake.Tensor):
            samples = tqdm(samples) if progressbar else samples
            for sample in samples:
                self._extend(
                    [sample],
                    update_commit_diff=update_commit_diff,
                    progressbar=False,
                    pg_callback=pg_callback,
                )  # TODO optimize this
            return samples
        if len(samples) == 0:
            return samples
        self._link_tensor_to_samples(samples)
        verified_samples = verified_samples or self.check_each_sample(
            samples, ignore_errors=ignore_errors
        )
        samples = self._sanitize_samples(samples, ignore_errors=ignore_errors)
        samples = self._samples_to_chunks(
            samples,
            start_chunk=self.last_appended_chunk(allow_copy=False),
            register=True,
            progressbar=progressbar,
            update_commit_diff=update_commit_diff,
            pg_callback=pg_callback,
            return_samples=True,
            ignore_errors=ignore_errors,
        )
        verified_samples = verified_samples or samples
        return verified_samples

    def _extend_link_callback(
        self, link_callback, samples, flat, progressbar, ignore_errors
    ):
        skipped = 0
        try:
            link_callback(samples, flat=flat, progressbar=progressbar)
        except Exception:
            if ignore_errors and not flat:
                # retry one at a time
                # don't retry if flat
                for i, sample in enumerate(samples):
                    try:
                        link_callback([sample], flat=flat, progressbar=progressbar)
                    except Exception:
                        # if link callback fails, remove the sample
                        self.pop(self.tensor_length - len(samples) + i - skipped)
                        skipped += 1
                return
            raise

    def _extend_sequence(
        self, samples, progressbar, link_callback, ignore_errors, verified_samples
    ):
        samples = tqdm(samples) if progressbar else samples
        already_verified = verified_samples is not None
        if not already_verified:
            verified_samples = []
        num_samples_added = 0
        for sample in samples:
            try:
                if sample is None:
                    sample = []
                self._link_tensor_to_samples(sample)
                if not already_verified:
                    verified_sample = self.check_each_sample(
                        sample, ignore_errors=ignore_errors
                    )
                sample = self._extend(
                    sample, progressbar=False, update_commit_diff=False
                )
                if not already_verified:
                    verified_sample = verified_sample or sample
                    verified_samples.append(
                        verified_sample if verified_sample is not None else sample
                    )
                self.sequence_encoder.register_samples(len(sample), 1)
                self.commit_diff.add_data(1)
                num_samples_added += 1
            except Exception:
                if ignore_errors:
                    continue
                raise

        if link_callback:
            skipped = []
            for i, s in enumerate(verified_samples):
                try:
                    self._extend_link_callback(
                        link_callback, s, True, progressbar, ignore_errors
                    )
                except Exception:
                    if ignore_errors:
                        self.pop(
                            self.tensor_length
                            - len(verified_samples)
                            + i
                            - len(skipped)
                        )
                        skipped.append(i)
                        continue
                    raise

            for i in reversed(skipped):
                verified_samples.pop(i)

            self._extend_link_callback(
                link_callback, verified_samples, False, progressbar, ignore_errors
            )

            # TODO: Handle case of samples passing the flat link callbacks
            # but failing the non-flat link callback but this is not yet a possibility

    def _prepare_samples_for_link_callback(self, samples):
        if not isinstance(samples, np.ndarray):
            samples = [
                (
                    None
                    if is_empty_list(s)
                    or (
                        isinstance(s, deeplake.core.tensor.Tensor) and s.is_empty_tensor
                    )
                    else s
                )
                for s in samples
            ]
        return samples

    def extend(
        self,
        samples,
        progressbar: bool = False,
        link_callback: Optional[Callable] = None,
        pg_callback=None,
        ignore_errors: bool = False,
        verified_samples=None,
    ):
        try:
            assert not (progressbar and pg_callback)
            self.check_link_ready()
            if not self.write_initialization_done:
                self._write_initialization()
                self.write_initialization_done = True

            initial_autoflush = self.cache.autoflush
            self.cache.autoflush = False
            num_samples = self.tensor_length

            if self.is_sequence:
                self._extend_sequence(
                    samples,
                    progressbar,
                    link_callback,
                    ignore_errors,
                    verified_samples,
                )
            else:
                verified_samples = self._extend(
                    samples,
                    progressbar,
                    pg_callback=pg_callback,
                    ignore_errors=ignore_errors,
                    verified_samples=verified_samples,
                )
                if link_callback:
                    verified_samples = self._prepare_samples_for_link_callback(
                        verified_samples
                    )
                    self._extend_link_callback(
                        link_callback,
                        verified_samples,
                        None,
                        progressbar,
                        ignore_errors,
                    )

            self.cache.autoflush = initial_autoflush
            self.cache.maybe_flush()
        except Exception as e:
            self.pop(list(range(num_samples, self.tensor_length)))
            raise SampleAppendError(self.name) from e

    def _create_new_chunk(self, register=True, row: Optional[int] = None) -> BaseChunk:
        """Creates and returns a new `Chunk`. Automatically creates an ID for it and puts a reference in the cache."""
        chunk_id = self.chunk_id_encoder.generate_chunk_id(register=register, row=row)
        chunk = self.chunk_class(*self.chunk_args)  # type: ignore
        chunk_name = ChunkIdEncoder.name_from_id(chunk_id)  # type: ignore
        chunk_key = get_chunk_key(self.key, chunk_name, self.commit_id)
        if self.commit_chunk_map is not None:
            self.commit_chunk_map.add(chunk_name)
        chunk.key = chunk_key
        chunk.id = chunk_id
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
        new_chunk_ids, tiles = self._samples_to_chunks(
            [sample], start_chunk=None, register=False
        )
        self.chunk_id_encoder._replace_chunks_for_tiled_sample(
            global_sample_index, new_chunk_ids
        )
        if tiles:
            self.tile_encoder.entries[global_sample_index] = tiles[0]
        else:
            del self.tile_encoder.entries[global_sample_index]

    def _update_tiled_sample(
        self, global_sample_index: int, index: Index, sample, nbytes_after_updates
    ):
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
            ).read_sample(0, is_tile=True),
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

    def _update_non_tiled_sample(
        self, global_sample_index: int, index: Index, sample, nbytes_after_updates
    ):
        enc = self.chunk_id_encoder
        chunk = self.get_chunks_for_sample(global_sample_index, copy=True)[0]
        local_sample_index = enc.translate_index_relative_to_chunks(global_sample_index)

        if len(index.values) <= 1 + int(self.is_sequence):
            chunk.update_sample(local_sample_index, sample)
        else:
            orig_sample = chunk.read_sample(local_sample_index, copy=True)
            sample = np.array(sample)
            lhs = orig_sample[tuple(e.value for e in index.values[1:])]
            if lhs.ndim > sample.ndim:
                sample = np.expand_dims(sample, tuple(range(sample.ndim, lhs.ndim)))
            lhs[:] = sample
            chunk.update_sample(local_sample_index, orig_sample)
        if (
            self.active_updated_chunk is not None
            and self.active_updated_chunk.key != chunk.key  # type: ignore
        ):
            self.write_chunk_to_storage(self.active_updated_chunk)
        self.active_updated_chunk = chunk

        # only care about deltas if it isn't the last chunk
        if chunk.key != self.last_chunk_key:  # type: ignore
            nbytes_after_updates.append(chunk.nbytes)

        self.pad_encoder.unpad(global_sample_index)

        self._check_rechunk(
            chunk, chunk_row=enc.__getitem__(global_sample_index, True)[0][1]
        )

    def pad_and_append(
        self,
        num_samples_to_pad: int,
        value,
        extend_link_callback=None,
        update_link_callback=None,
    ):
        """Pads the tensor with empty samples and appends value at the end."""
        self.check_link_ready()
        self.start_chunk = self.last_appended_chunk()  # type: ignore
        update_first_sample = False
        num_samples = self.num_samples
        orig_num_samples_to_pad = num_samples_to_pad
        if num_samples_to_pad > 0:
            if num_samples == 0:
                # set htype, dtype, shape, we later update it with empty sample
                self.extend([value], link_callback=extend_link_callback)
                num_samples_to_pad -= 1
                update_first_sample = True
            htype = self.tensor_meta.htype
            if htype in ("json", "text", "list"):
                empty_sample = get_empty_text_like_sample(htype)
                empty_samples = [empty_sample] * num_samples_to_pad
            elif self.tensor_meta.is_link:
                empty_sample = None
                empty_samples = [None] * num_samples_to_pad
            else:
                ndim = len(self.tensor_meta.max_shape)
                if self.is_sequence:
                    ndim += 1
                shape = tuple([num_samples_to_pad] + [0] * ndim)
                dtype = self.tensor_meta.dtype
                empty_sample = np.zeros(shape[1:], dtype=dtype)
                empty_samples = np.zeros(shape, dtype=dtype)  # type: ignore

            if update_first_sample:
                self.update(Index(0), empty_sample, link_callback=update_link_callback)
            # pad
            self.extend(empty_samples, link_callback=extend_link_callback)
            self.pad_encoder.add_padding(num_samples, orig_num_samples_to_pad)
        self.extend([value], link_callback=extend_link_callback)

    def update(
        self,
        index: Index,
        samples: Union[np.ndarray, Sequence[InputSample], InputSample],
        operator: Optional[str] = None,
        link_callback: Optional[Callable] = None,
    ):
        """Update data at `index` with `samples`."""

        cmap = self.commit_chunk_map
        if cmap is not None:
            cmap = CommitChunkMap.frombuffer(cmap.tobytes())
        try:
            self.check_link_ready()
            (self._sequence_update if self.is_sequence else self._update)(  # type: ignore
                index,
                samples,
                operator,
                link_callback=link_callback,
            )
        except Exception as e:
            if cmap is not None:
                key = get_tensor_commit_chunk_map_key(self.key, self.commit_id)
                self.meta_cache[key] = cmap
                self._commit_chunk_map = cmap
                self.meta_cache.register_deeplake_object(key, cmap)
            raise SampleUpdateError(self.name) from e

    def _get_samples_to_move(self, chunk) -> List[Sample]:
        decompress = isinstance(chunk, ChunkCompressedChunk) or self.is_text_like
        samples_to_move: List[Sample] = []
        sum_bytes = 0

        for idx in range(chunk.num_samples - 1, 1, -1):
            sample_data = chunk.read_sample(idx, decompress=decompress)
            sum_bytes += len(sample_data)
            if sum_bytes > int(RANDOM_MAX_ALLOWED_CHUNK_SIZE / 2):
                break
            sample_shape = chunk.shapes_encoder[idx]
            new_sample = self._get_sample_object(
                sample_data, sample_shape, chunk.compression, chunk.dtype, decompress
            )
            samples_to_move.append(new_sample)
        samples_to_move.reverse()
        return samples_to_move

    def _get_chunk_samples(self, chunk) -> List[Optional[Sample]]:
        decompress = isinstance(chunk, ChunkCompressedChunk) or self.is_text_like
        all_samples_in_chunk: List[Optional[Sample]] = []

        for idx in range(chunk.num_samples):
            sample_data = chunk.read_sample(idx, decompress=decompress)
            try:
                sample_shape = chunk.shapes_encoder[idx]
            except IndexError:
                all_samples_in_chunk.append(None)
                continue
            new_sample = self._get_sample_object(
                sample_data, sample_shape, chunk.compression, chunk.dtype, decompress
            )
            all_samples_in_chunk.append(new_sample)

        return all_samples_in_chunk

    def _get_sample_object(
        self, sample_data, sample_shape, compression, dtype, decompress
    ):
        if isinstance(sample_data, Polygons):
            return sample_data

        if self.is_text_like:
            if self.tensor_meta.is_link:
                sample = LinkedSample(sample_data)
            else:
                sample = sample_data
                if self.tensor_meta.htype == "json" and isinstance(sample, np.ndarray):
                    sample = sample.squeeze()
            return sample

        if decompress:
            sample = Sample(array=sample_data, shape=sample_shape)
        else:
            # sample data should not be an array here
            assert not isinstance(sample_data, np.ndarray)
            sample = Sample(
                buffer=sample_data,
                shape=sample_shape,
                compression=compression,
                dtype=dtype,
            )
        return sample

    def __rechunk(self, chunk: BaseChunk, chunk_row: int):
        samples_to_move = self._get_samples_to_move(chunk=chunk)
        num_samples = len(samples_to_move)
        if num_samples == 0:
            return
        new_chunk = self._create_new_chunk(register=True, row=chunk_row)
        new_chunk_row = chunk_row + 1

        self.chunk_id_encoder.decrease_samples(row=chunk_row, num_samples=num_samples)
        self.chunk_id_encoder.decrease_samples(
            row=new_chunk_row, num_samples=num_samples
        )
        chunk.pop_multiple(num_samples=len(samples_to_move))
        samples = self._sanitize_samples(samples_to_move)
        self._samples_to_chunks(
            samples,
            start_chunk=new_chunk,
            register=True,
            update_commit_diff=False,
            update_tensor_meta=False,
            start_chunk_row=new_chunk_row,
            register_creds=False,
        )

    def _merge_chunks(
        self,
        from_chunk: BaseChunk,
        from_chunk_row: int,
        to_chunk: BaseChunk,
        to_chunk_row: int,
    ):
        samples_to_move = self._get_chunk_samples(chunk=from_chunk)
        num_samples = len(samples_to_move)
        if num_samples == 0:
            return True

        from_chunk.pop_multiple(num_samples=num_samples)
        samples = self._sanitize_samples(samples_to_move)
        to_chunk.is_dirty = True
        self.active_updated_chunk = to_chunk
        self._samples_to_chunks(
            samples,
            start_chunk=to_chunk,
            register=True,
            update_commit_diff=False,  # merging chunks should not update diff
            update_tensor_meta=False,
            start_chunk_row=to_chunk_row,
            register_creds=False,
        )
        self.chunk_id_encoder.delete_chunk_id(row=from_chunk_row)
        try:
            del self.cache[from_chunk.key]  # type: ignore
        except KeyError:
            pass
        self.cache[to_chunk.key] = to_chunk  # type: ignore
        return True

    def _is_tiled(self, row: int) -> bool:
        """checkes whether the chunk is tiled or not

        Args:
            row (int): Represents the row of the chunk.

        Returns:
            bool: return true if the current chunk and previous/next row chunk have the same chunk index false otherwise.
        """

        arr = self.chunk_id_encoder.array
        if row >= 1 and len(arr) > 1:
            if arr[row][LAST_SEEN_INDEX_COLUMN] == arr[row - 1][LAST_SEEN_INDEX_COLUMN]:
                return True
        if len(arr) > row + 1:
            if arr[row][LAST_SEEN_INDEX_COLUMN] == arr[row + 1][LAST_SEEN_INDEX_COLUMN]:
                return True
        return False

    def _try_merge_with_next_chunk(self, chunk: BaseChunk, row: int) -> bool:
        next_chunk_id = self.chunk_id_encoder.get_next_chunk_id(row)
        if next_chunk_id is None:
            return False
        next_chunk_row = row + 1
        if self._is_tiled(next_chunk_row):
            return False

        next_chunk_name = ChunkIdEncoder.name_from_id(next_chunk_id)  # type: ignore
        next_chunk_commit_id, tkey = self.get_chunk_commit(next_chunk_name)
        chunk_key = get_chunk_key(tkey, next_chunk_name, next_chunk_commit_id)
        next_chunk_size = self.cache.get_object_size(chunk_key)
        next_chunk = self.get_chunk_from_chunk_id(int(next_chunk_id))
        if next_chunk_size + chunk.num_data_bytes < next_chunk.min_chunk_size:
            if next_chunk_commit_id != self.commit_id:
                next_chunk = self.copy_chunk_to_new_commit(next_chunk, next_chunk_name)
            chunk_id = chunk.id
            chunk_name = ChunkIdEncoder.name_from_id(chunk_id)
            chunk_commit_id, tkey = self.get_chunk_commit(chunk_name)
            if chunk_commit_id != self.commit_id:
                chunk = self.copy_chunk_to_new_commit(chunk, chunk_name)
            return self._merge_chunks(
                from_chunk=next_chunk,
                from_chunk_row=next_chunk_row,
                to_chunk=chunk,
                to_chunk_row=row,
            )
        return False

    def _try_merge_with_previous_chunk(self, chunk: BaseChunk, row: int) -> bool:
        prev_chunk_id = self.chunk_id_encoder.get_prev_chunk_id(row)
        if prev_chunk_id is None:
            return False

        prev_chunk_row = row - 1
        if self._is_tiled(prev_chunk_row):
            return False

        prev_chunk_name = ChunkIdEncoder.name_from_id(prev_chunk_id)  # type: ignore
        prev_chunk_commit_id, tkey = self.get_chunk_commit(prev_chunk_name)
        prev_chunk_key = get_chunk_key(tkey, prev_chunk_name, prev_chunk_commit_id)
        prev_chunk_size = self.cache.get_object_size(prev_chunk_key)
        prev_chunk = self.get_chunk_from_chunk_id(int(prev_chunk_id))
        if prev_chunk_size + chunk.num_data_bytes < prev_chunk.min_chunk_size:
            if prev_chunk_commit_id != self.commit_id:
                prev_chunk = self.copy_chunk_to_new_commit(prev_chunk, prev_chunk_name)
            chunk_id = chunk.id
            chunk_name = ChunkIdEncoder.name_from_id(chunk_id)
            chunk_commit_id, tkey = self.get_chunk_commit(chunk_name)
            if chunk_commit_id != self.commit_id:
                chunk = self.copy_chunk_to_new_commit(chunk, chunk_name)
            # merge with previous chunk
            return self._merge_chunks(
                from_chunk=chunk,
                from_chunk_row=row,
                to_chunk=prev_chunk,
                to_chunk_row=prev_chunk_row,
            )
        return False

    def _try_merge_with_neighbor_and_split(self, chunk: BaseChunk, row: int):
        if self._try_merge_with_previous_chunk(chunk, row) is False:
            self._try_merge_with_next_chunk(chunk, row)

    def is_tensor_hidden(self) -> bool:
        """function to check is the tensors that chunk_engine belongs to is hidden"""
        tensor_name = self.tensor_meta.name or self.key
        if tensor_name.startswith("_"):
            return (
                tensor_name.endswith("_shape")
                or tensor_name.endswith("_id")
                or tensor_name.endswith("_info")
            )
        return False

    def _check_rechunk(self, chunk: BaseChunk, chunk_row: int):
        """function to check if there is a need to re-chunk the current one"""

        if self.is_tensor_hidden():
            return
        if (
            chunk.num_data_bytes < RANDOM_MINIMAL_CHUNK_SIZE
            and self.max_chunk_size > RANDOM_MINIMAL_CHUNK_SIZE
        ):
            self._try_merge_with_neighbor_and_split(chunk=chunk, row=chunk_row)

        elif (
            chunk.num_data_bytes > RANDOM_MAX_ALLOWED_CHUNK_SIZE
            or chunk.num_data_bytes > self.max_chunk_size + RANDOM_MINIMAL_CHUNK_SIZE
        ):
            self.__rechunk(chunk, chunk_row)

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
        try:
            if operator is not None:
                return self._update_with_operator(index, samples, operator)

            enc = self.chunk_id_encoder
            index_length = index.length(self.num_samples)
            samples = make_sequence(samples, index_length)
            self._link_tensor_to_samples(samples)
            verified_samples = self.check_each_sample(samples)
            if self.tensor_meta.htype == "class_label":
                samples = self._convert_class_labels(samples)
            if self.tensor_meta.htype == "polygon":
                samples = [Polygons(sample, self.tensor_meta.dtype) for sample in samples]  # type: ignore
            nbytes_after_updates: List[int] = []
            global_sample_indices = tuple(index.values[0].indices(self.num_samples))
            is_sequence = self.is_sequence
            for i, sample in enumerate(samples):  # type: ignore
                sample = None if is_empty_list(sample) else sample
                global_sample_index = global_sample_indices[i]  # TODO!
                if self._is_tiled_sample(global_sample_index):
                    self._update_tiled_sample(
                        global_sample_index, index, sample, nbytes_after_updates
                    )
                else:
                    self._update_non_tiled_sample(
                        global_sample_index, index, sample, nbytes_after_updates
                    )
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
        finally:
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
            if isinstance(samples, deeplake.core.tensor.Tensor):
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
        self._update(index1, arr)

    def read_bytes_for_sample(self, global_sample_index: int) -> bytes:
        if self.chunk_compression:
            raise ValueError(
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
        if self.is_sequence:
            assert self.sequence_encoder is not None
            start_idx, end_idx = self.sequence_encoder[global_sample_index]
            end_idx -= 1
            start_idx, end_idx = map(
                enc.translate_index_relative_to_chunks, (start_idx, end_idx)
            )
            sb = chunk.byte_positions_encoder[start_idx][0]
            eb = chunk.byte_positions_encoder[end_idx][1]
        else:
            local_sample_index = enc.translate_index_relative_to_chunks(
                global_sample_index
            )
            sb, eb = chunk.byte_positions_encoder[local_sample_index]
        return buffer[sb:eb].tobytes()

    def read_shape_for_sample(
        self,
        global_sample_index: int,
    ) -> Tuple[int, ...]:
        enc = self.chunk_id_encoder
        if self._is_tiled_sample(global_sample_index):
            return self.tile_encoder.get_sample_shape(global_sample_index)
        local_sample_index = enc.translate_index_relative_to_chunks(global_sample_index)
        if self.is_video:
            chunk_id = enc[global_sample_index][0]
            chunk = self.get_video_chunk(chunk_id)[0]
        else:
            chunk_id, _, worst_case_header_size = self.get_chunk_info(
                global_sample_index, fetch_chunks=False
            )
            chunk = self.get_chunk_from_chunk_id(
                chunk_id, partial_chunk_bytes=worst_case_header_size
            )
        return tuple(map(int, chunk.shapes_encoder[local_sample_index]))

    @property
    def is_fixed_shape(self):
        tensor_meta = self.tensor_meta
        return not self.is_text_like and tensor_meta.min_shape == tensor_meta.max_shape

    @property
    def num_samples_per_chunk(self):
        # should only be called if self.is_fixed_shape
        if self._num_samples_per_chunk is None:
            self._num_samples_per_chunk = int(
                self.chunk_id_encoder.array[0, LAST_SEEN_INDEX_COLUMN] + 1
            )
        return self._num_samples_per_chunk

    def read_sample_from_chunk(
        self,
        global_sample_index: int,
        chunk: BaseChunk,
        cast: bool = True,
        copy: bool = False,
        decompress: bool = True,
        to_pil: bool = False,
    ) -> Union[np.ndarray, Image.Image]:
        enc = self.chunk_id_encoder
        if self.is_fixed_shape and self.sample_compression is None:
            num_samples_per_chunk = self.num_samples_per_chunk
            local_sample_index = global_sample_index % num_samples_per_chunk
        else:
            local_sample_index = enc.translate_index_relative_to_chunks(
                global_sample_index
            )
        if to_pil:
            assert isinstance(chunk, SampleCompressedChunk)
            return chunk.read_sample(
                local_sample_index,
                cast=cast,
                copy=copy,
                decompress=decompress,
                to_pil=True,
            )

        return chunk.read_sample(
            local_sample_index, cast=cast, copy=copy, decompress=decompress
        )

    def _get_full_chunk(self, index) -> bool:
        """Reads samples from chunks and returns as a boolean that says whether we need to fetch full chunks or only specified subset of it.
        Args:
            index (Index): Represents the samples to read from chunks. See `Index` for more information.
        Returns:
            bool: True/False, whether to fetch a full chunk or only a part of it.
        """
        threshold = 10

        if type(index.values[0].value) == slice:
            start = index.values[0].value.start or 0
            stop = index.values[0].value.stop or self.num_samples
            step = index.values[0].value.step or 1

            if start < 0:
                start = self.num_samples + start

            if stop < 0:
                stop = self.num_samples + stop

            numpy_array_length = (stop - start) // step
            return numpy_array_length > threshold
        return False

    def numpy(
        self,
        index: Index,
        aslist: bool = False,
        use_data_cache: bool = True,
        fetch_chunks: bool = False,
        pad_tensor: bool = False,
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """Reads samples from chunks and returns as a numpy array. If `aslist=True`, returns a sequence of numpy arrays.

        Args:
            index (Index): Represents the samples to read from chunks. See `Index` for more information.
            aslist (bool): If True, the samples will be returned as a list of numpy arrays. If False, returns a single numpy array. Defaults to False.
            use_data_cache (bool): If True, the data cache is used to speed up the read if possible. If False, the data cache is ignored. Defaults to True.
            fetch_chunks (bool): If True, full chunks will be retrieved from the storage, otherwise only required bytes will be retrieved.
                This will always be True even if specified as False in the following cases:
                - The tensor is ChunkCompressed
                - The chunk which is being accessed has more than 128 samples.
            pad_tensor (bool): If True, any index out of bounds will not throw an error, but instead will return an empty sample.

        Raises:
            DynamicTensorNumpyError: If shapes of the samples being read are not all the same.

        Returns:
            Union[np.ndarray, List[np.ndarray]]: Either a list of numpy arrays or a single numpy array (depending on the `aslist` argument).

        Note:
            For polygons, ``aslist`` is always ``True``.
        """
        self.check_link_ready()
        fetch_chunks = fetch_chunks or self._get_full_chunk(index)
        return (self._sequence_numpy if self.is_sequence else self._numpy)(
            index, aslist, use_data_cache, fetch_chunks, pad_tensor
        )

    def get_chunk_info(self, global_sample_index: int, fetch_chunks: bool):
        """Returns the chunk_id, row and worst case header size of chunk containing the given sample."""
        enc = self.chunk_id_encoder
        out = enc.__getitem__(global_sample_index, return_row_index=True)  # type: ignore
        chunk_id, row = out[0][0], out[0][1]

        worst_case_header_size = 0
        num_samples_in_chunk = -1
        if (
            not fetch_chunks
            and self.chunk_class != ChunkCompressedChunk
            and isinstance(self.base_storage, (S3Provider, GCSProvider, AzureProvider))
        ):
            prev = int(enc.array[row - 1][LAST_SEEN_INDEX_COLUMN]) if row > 0 else -1
            num_samples_in_chunk = int(enc.array[row][LAST_SEEN_INDEX_COLUMN]) - prev
            worst_case_header_size += HEADER_SIZE_BYTES + 10  # 10 for version
            ENTRY_SIZE = 4
            if self.tensor_meta.max_shape == self.tensor_meta.min_shape:
                num_shape_entries = 1 * (len(self.tensor_meta.min_shape) + 1)
                if self.is_text_like:
                    num_bytes_entries = num_samples_in_chunk * 3
                elif self.sample_compression is None:
                    num_bytes_entries = 1 * 3
                else:
                    num_bytes_entries = num_samples_in_chunk * 3
            else:
                num_shape_entries = num_samples_in_chunk * (
                    1 + len(self.tensor_meta.max_shape)
                )
                num_bytes_entries = num_samples_in_chunk * 3
            bytes_enc_size = num_bytes_entries * ENTRY_SIZE
            shape_enc_size = num_shape_entries * ENTRY_SIZE
            worst_case_header_size += shape_enc_size
            worst_case_header_size += bytes_enc_size

        return chunk_id, row, worst_case_header_size

    def translate_to_local_index(self, global_sample_index: int, row: int):
        """Translate global sample index to local index relative to chunks without another encoder lookup."""
        if row == 0:
            return global_sample_index
        return global_sample_index - (
            self.chunk_id_encoder.array[row - 1][-1].item() + 1
        )

    def read_video_sample_from_chunk(
        self,
        chunk_id: int,
        local_sample_index: int,
        index: Index,
        decompress: bool = True,
    ):
        assert self.is_video
        chunk, stream = self.get_video_chunk(chunk_id)
        sub_index = index.values[1].value if len(index.values) > 1 else None  # type: ignore
        sample = chunk.read_sample(
            local_sample_index,
            sub_index=sub_index,
            stream=stream,
            decompress=decompress,
        )
        if decompress:
            return sample[tuple(entry.value for entry in index.values[2:])]
        return sample

    def read_basic_sample_from_chunk(
        self,
        chunk_id: int,
        local_sample_index: int,
        index: Index,
        worst_case_header_size: int = 0,
        is_tile: bool = False,
        decompress: bool = True,
    ):
        chunk = self.get_chunk_from_chunk_id(
            chunk_id, partial_chunk_bytes=worst_case_header_size
        )
        decompress = decompress or (
            isinstance(chunk, ChunkCompressedChunk) or len(index) > 1
        )
        ret = chunk.read_sample(
            local_sample_index,
            cast=self.tensor_meta.htype != "dicom",
            is_tile=is_tile,
            decompress=decompress,
        )
        if len(index) > 1:
            ret = ret[tuple(entry.value for entry in index.values[1:])]
        return ret

    def get_basic_sample(
        self,
        global_sample_index: int,
        index: Index,
        fetch_chunks: bool = False,
        is_tile: bool = False,
        decompress: bool = True,
    ):
        chunk_id, row, worst_case_header_size = self.get_chunk_info(
            global_sample_index, fetch_chunks=fetch_chunks
        )
        local_sample_index = self.translate_to_local_index(global_sample_index, row)
        return self.read_basic_sample_from_chunk(
            chunk_id,
            local_sample_index,
            index,
            worst_case_header_size,
            is_tile=is_tile,
            decompress=decompress,
        )

    def get_video_sample(
        self, global_sample_index: int, index: Index, decompress: bool = True
    ):
        assert self.is_video
        chunk_id, row, _ = self.get_chunk_info(global_sample_index, fetch_chunks=True)
        local_sample_index = self.translate_to_local_index(global_sample_index, row)
        return self.read_video_sample_from_chunk(
            chunk_id, local_sample_index, index, decompress=decompress
        )

    def get_non_tiled_sample(
        self, global_sample_index, index, fetch_chunks=False, decompress=True
    ):
        if self.is_video:
            return self.get_video_sample(global_sample_index, index, decompress)
        return self.get_basic_sample(
            global_sample_index,
            index,
            fetch_chunks,
            is_tile=False,
            decompress=decompress,
        )

    def get_full_tiled_sample(self, global_sample_index: int):
        chunks = self.get_chunks_for_sample(global_sample_index)
        return combine_chunks(chunks, global_sample_index, self.tile_encoder)

    def get_partial_tiled_sample(self, global_sample_index: int, index: Index):
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
            lambda chunk_id: self.get_chunk_from_chunk_id(chunk_id).read_sample(
                0, is_tile=True
            ),
            otypes=[object],
        )(required_tile_ids)
        sample = coalesce_tiles(tiles, tile_shape, None, self.tensor_meta.dtype)
        sample = sample[sample_index]
        return sample

    def get_single_sample(
        self,
        global_sample_index: int,
        index: Index,
        fetch_chunks: bool = False,
        pad_tensor: bool = False,
        decompress: bool = True,
    ):
        if pad_tensor and global_sample_index >= self.tensor_length:
            return self.get_empty_sample(index)

        if not self._is_tiled_sample(global_sample_index):
            sample = self.get_non_tiled_sample(
                global_sample_index,
                index,
                fetch_chunks=fetch_chunks,
                decompress=decompress,
            )
        elif len(index.values) == 1:
            sample = self.get_full_tiled_sample(global_sample_index)
        else:
            sample = self.get_partial_tiled_sample(global_sample_index, index)

        return sample

    def _load_chunk(
        self,
        chunk_info: Tuple[str, int, List[int], bool],
        storages: Dict[int, StorageProvider],
    ):
        """Worker function for chunk retrieval."""
        chunk_key = self.get_chunk_key_for_id(chunk_info[0])
        result = self.cache._get_item_from_cache(chunk_key)
        if result is not None:
            return result, chunk_info
        is_tile = chunk_info[3]
        if is_tile:
            return None, chunk_info
        cache_used_percent = lambda: self.cache.cache_used / self.cache.cache_size
        while cache_used_percent() > 0.9:
            self.cache._pop_from_cache()
        base_storage = storages.get(threading.get_ident())
        if base_storage is None:
            if isinstance(self.base_storage, MemoryProvider):
                base_storage = self.base_storage
            else:
                base_storage = self.base_storage.copy()
            storages[threading.get_ident()] = base_storage
        chunk = base_storage.__getitem__(chunk_key)
        return chunk, chunk_info

    def _get_chunk_infos(self, indices: List[int]):
        """Returns chunk infos for the chunks covered by the given indices."""
        indices = sorted(indices)
        indices = np.asarray(indices, dtype=np.uint32)  # type: ignore
        encoded = self.chunk_id_encoder._encoded
        last_idxs = encoded[:, -1]

        pos = np.searchsorted(indices, last_idxs, side="right")

        chunk_infos: List[List[Union[int, int, List[int], bool]]] = []

        last_pos = 0
        for i in range(len(last_idxs)):
            is_tile = False

            if pos[i] == 0:
                continue

            if pos[i] == last_pos:
                # not tile
                if last_idxs[i] != last_idxs[i - 1]:
                    if pos[i] == len(indices):
                        break
                    continue
                # mark the previous chunk as tile
                chunk_infos[-1][3] = True
                # mark this chunk as tile
                is_tile = True

            idxs_in_chunk = indices[last_pos : pos[i]].tolist()  # type: ignore

            last_pos = pos[i]

            chunk_id = encoded[i][0].item()
            row = i

            chunk_infos.append([chunk_id, row, idxs_in_chunk, is_tile])

        return chunk_infos

    def load_chunks(
        self,
        indices: List[int],
        in_order: bool = False,
        reverse: bool = False,
    ):
        """Fetches relevant chunks from base storage, adds them to cache and yields chunk info.
        If ``in_order`` is ``True``, chunks are yielded in order of the chunk_id_encoder.
        If ``reverse`` is ``True``, chunks are yielded in reverse order of the chunk_id_encoder.
        """
        chunk_infos = self._get_chunk_infos(indices)

        # some storage providers are not thread safe
        storages: Dict[int, StorageProvider] = {}

        if not (in_order or reverse):
            with ThreadPoolExecutor() as executor:
                futures_list = [
                    executor.submit(self._load_chunk, chunk_info, storages)
                    for chunk_info in chunk_infos
                ]
                for future in futures.as_completed(futures_list):
                    exception = future.exception()
                    if exception:
                        raise exception
                    chunk, chunk_info = future.result()
                    if chunk:
                        if _get_nbytes(chunk) <= self.cache.cache_size:
                            self.cache._insert_in_cache(
                                self.get_chunk_key_for_id(chunk_info[0]), chunk
                            )
                    yield chunk_info
        else:
            with ThreadPoolExecutor() as executor:
                for result in executor.map(
                    self._load_chunk,
                    reversed(chunk_infos) if reverse else chunk_infos,
                    repeat(storages),
                ):
                    chunk, chunk_info = result
                    if chunk:
                        if _get_nbytes(chunk) <= self.cache.cache_size:
                            self.cache._insert_in_cache(
                                self.get_chunk_key_for_id(chunk_info[0]), chunk
                            )
                    yield chunk_info

    def _get_samples(
        self,
        chunk_id: int,
        row: int,
        idxs: List[int],
        index: Index,
        is_polygon: bool,
        aslist: bool,
        pad_tensor: bool,
    ):
        """Get samples from a chunk.

        Args:
            chunk_id (int): Chunk to read samples from. Can be ``None`` in case of tiles.
            row (int): Row of the chunk in the chunk_id_encoder.
            idxs (List[int]): List of global sample indices to read from this chunk.
            index (Index): Original index applied on the tensor.
            is_polygon (bool): Whether the tensor is a polygon tensor.
            aslist (bool): Whether to return a list or numpy array.
            pad_tensor (bool): Whether tensor is padded.

        Raises:
            GetChunkError: If a chunk cannot be retrieved from the storage.
            ReadSampleFromChunkError: If a sample cannot be read from a chunk.

        Returns:
            Dict of samples and shape of the last sample encountered in this chunk.
        """
        samples = {}
        last_shape = None

        for idx in idxs:
            if idx in samples:
                continue
            try:
                if not self._is_tiled_sample(idx) and idx < self.num_samples:
                    local_idx = self.translate_to_local_index(idx, row)
                    sample = self.read_basic_sample_from_chunk(
                        chunk_id, local_idx, index
                    )
                else:
                    sample = self.get_single_sample(idx, index, pad_tensor=pad_tensor)
            except GetChunkError as e:
                raise GetChunkError(e.chunk_key, idx, self.name, e) from e
            except ReadSampleFromChunkError as e:
                raise ReadSampleFromChunkError(e.chunk_key, idx, self.name) from e
            check_sample_shape(sample.shape, last_shape, self.key, index, aslist)
            last_shape = sample.shape
            if is_polygon:
                sample = [p.__array__() for p in sample]
            samples[idx] = sample
        return samples, last_shape

    def get_samples(self, index: Index, aslist: bool, pad_tensor: bool):
        """Get samples for the given index, fetches chunks in parallel.

        Args:
            index (Index): Index applied on the tensor.
            aslist (bool): Whether to return a list or numpy array.
            pad_tensor (bool): Whether tensor is padded.

        Returns:
            List of samples.
        """
        last_shape = None
        is_polygon = self.tensor_meta.htype == "polygon"
        read_samples = partial(
            self._get_samples,
            index=index,
            is_polygon=is_polygon,
            aslist=aslist,
            pad_tensor=pad_tensor,
        )
        samples = {}
        for chunk_id, row, idxs, is_tile in self.load_chunks(
            list(index.values[0].indices(self.num_samples))
        ):
            chunk_samples, chunk_last_shape = read_samples(chunk_id, row, idxs)
            check_sample_shape(chunk_last_shape, last_shape, self.key, index, aslist)
            samples.update(chunk_samples)
            cache_used_percent = lambda: self.cache.cache_used / self.cache.cache_size
            while cache_used_percent() > 0.9:
                self.cache._pop_from_cache()

        return [samples[idx] for idx in index.values[0].indices(self.num_samples)]

    def _numpy(
        self,
        index: Index,
        aslist: bool = False,
        use_data_cache: bool = True,
        fetch_chunks: bool = False,
        pad_tensor: bool = False,
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """Reads samples from chunks and returns as a numpy array. If `aslist=True`, returns a sequence of numpy arrays.

        Args:
            index (Index): Represents the samples to read from chunks. See `Index` for more information.
            aslist (bool): If True, the samples will be returned as a list of numpy arrays. If False, returns a single numpy array. Defaults to False. For polygons, aslist is always True.
            use_data_cache (bool): If True, the data cache is used to speed up the read if possible. If False, the data cache is ignored. Defaults to True.
            fetch_chunks (bool): If True, full chunks will be retrieved from the storage, otherwise only required bytes will be retrieved.
                This will always be True even if specified as False in the following cases:
                - The tensor is ChunkCompressed
                - The chunk which is being accessed has more than 128 samples.
            pad_tensor (bool): If True, any index out of bounds will not throw an error, but instead will return an empty sample.

        Raises:
            DynamicTensorNumpyError: If shapes of the samples being read are not all the same.
            GetChunkError: If a chunk cannot be retrieved from the storage.
            ReadSampleFromChunkError: If a sample cannot be read from a chunk.
            GetDataFromLinkError: If data cannot be retrieved from a link.

        Returns:
            Union[np.ndarray, List[np.ndarray]]: Either a list of numpy arrays or a single numpy array (depending on the `aslist` argument).
        """
        length = self.num_samples
        last_shape = None
        ispolygon = self.tensor_meta.htype == "polygon"
        if ispolygon:
            aslist = True
        if use_data_cache and self.is_data_cachable:
            samples = self.numpy_from_data_cache(index, length, aslist, pad_tensor)
        else:
            samples = []
            if (
                fetch_chunks
                and not (self.tensor_meta.is_link or self.is_video)
                and not isinstance(self.base_storage, MemoryProvider)
            ):
                samples = self.get_samples(index, aslist, pad_tensor)
            else:
                for global_sample_index in index.values[0].indices(length):
                    try:
                        sample = self.get_single_sample(
                            global_sample_index,
                            index,
                            fetch_chunks=fetch_chunks,
                            pad_tensor=pad_tensor,
                        )
                    except GetChunkError as e:
                        raise GetChunkError(
                            e.chunk_key, global_sample_index, self.name, e
                        ) from e
                    except ReadSampleFromChunkError as e:
                        raise ReadSampleFromChunkError(
                            e.chunk_key, global_sample_index, self.name
                        ) from e
                    except GetDataFromLinkError as e:
                        raise GetDataFromLinkError(
                            e.link, global_sample_index, self.name
                        ) from e

                    check_sample_shape(
                        sample.shape, last_shape, self.key, index, aslist
                    )
                    last_shape = sample.shape
                    if ispolygon:
                        sample = [p.__array__() for p in sample]
                    samples.append(sample)
        if aslist and all(map(np.isscalar, samples)):
            samples = list(arr.item() for arr in samples)

        if not index.values[0].subscriptable():
            samples = samples[0]

        if aslist:
            return samples
        return np.array(samples)

    def numpy_from_data_cache(self, index, length, aslist, pad_tensor=False):
        samples = []
        enc = self.chunk_id_encoder
        for global_sample_index in index.values[0].indices(length):
            if pad_tensor and global_sample_index >= self.tensor_length:
                sample = self.get_empty_sample()
                try:
                    sample = sample[tuple(entry.value for entry in index.values[1:])]
                except IndexError:
                    pass
            else:
                if (
                    self.cached_data is None
                    or global_sample_index not in self.cache_range
                ):
                    row = enc.__getitem__(global_sample_index, True)[0][1]
                    chunks = self.get_chunks_for_sample(global_sample_index)
                    assert len(chunks) == 1

                    chunk_arr = self.chunk_id_encoder.array

                    chunk = chunks[0]
                    first_sample = int(0 if row == 0 else chunk_arr[row - 1][1] + 1)
                    last_sample = int(self.chunk_id_encoder.array[row][1])
                    num_samples = last_sample - first_sample + 1
                    full_shape = (num_samples,) + tuple(self.tensor_meta.max_shape)
                    dtype = self.tensor_meta.dtype

                    data_bytes = bytearray(chunk.data_bytes)
                    self.cached_data = np.frombuffer(data_bytes, dtype).reshape(
                        full_shape
                    )
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
            self.get_chunk_from_chunk_id(chunk_id, copy)
            for chunk_id in self.chunk_id_encoder[global_sample_index]
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
            arr = self.chunk_id_encoder._encoded
            if not arr.size:
                return []
            return [
                ChunkIdEncoder.name_from_id(chunk_id)
                for chunk_id in self.chunk_id_encoder._encoded[:, CHUNK_ID_COLUMN]
            ]  # type: ignore
        else:
            return [k for (k, v) in self.commit_chunk_map.chunks.items() if not v]  # type: ignore

    def list_all_chunks_path(self) -> List[str]:
        """Return list of paths to all chunks"""
        commit_id = self.commit_id
        return [
            get_chunk_key(self.key, chunk, commit_id)
            for chunk in self.list_all_chunks()
        ]

    def list_orphaned_chunks(self, storage: StorageProvider) -> List[str]:
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

    def pop(
        self,
        indices: Optional[Union[int, List[int]]] = None,
        link_callback: Optional[Callable] = None,
        sample_id_tensor=None,
    ):
        """Pop samples from the tensor at the given indices.

        Args:
            indices (Optional[Union[int, List[int]]]): List of indices to pop.
            link_callback (Optional[Callable]): Callback function to be called after popping each sample. Defaults to None.
            sample_id_tensor (Optional[deeplake.Tensor]): Associated sample ID tensor, if any.
        """
        if indices is None:
            indices = [self.tensor_length - 1]

        if not isinstance(indices, list):
            indices = [indices]

        self._write_initialization()
        self.cached_data = None
        initial_autoflush = self.cache.autoflush
        self.cache.autoflush = False

        def update_links_and_encoders(idx):
            """Update linked tensors and sample level encoders"""
            self.commit_diff.pop(
                idx,
                (
                    sample_id_tensor[idx].numpy().item()
                    if sample_id_tensor is not None
                    else None
                ),
            )
            if link_callback:
                link_callback(idx)
            if self.is_sequence:
                self.sequence_encoder.pop(idx)
            self.pad_encoder.pop(idx)

        if self.is_sequence:
            assert self.sequence_encoder is not None
            item_lengths = [
                [index, -np.subtract(*self.sequence_encoder[index])]
                for index in sorted(indices)
            ]
            flat_indices: List[int] = []
            for index in indices:
                flat_indices.extend(range(*self.sequence_encoder[index]))
            indices = flat_indices

        for chunk_id, row, idxs, is_tile in self.load_chunks(indices, reverse=True):
            idxs = list(reversed(idxs))
            if self.is_sequence:
                num_flat_samples = len(idxs)
                while item_lengths and num_flat_samples >= item_lengths[-1][1]:
                    num_flat_samples -= item_lengths[-1][1]
                    idx_2d, _ = item_lengths.pop()
                    update_links_and_encoders(idx_2d)

                if num_flat_samples:
                    item_lengths[-1][1] -= num_flat_samples
            else:
                for idx in idxs:
                    update_links_and_encoders(idx)
            self.pop_samples(chunk_id, row, idxs, is_tile)

        self.cache.autoflush = initial_autoflush
        self.cache.maybe_flush()

    def _pop_from_chunk(self, chunk: Optional[BaseChunk], row: int, global_idx: int):
        """Pop sample from chunk. If chunk is ``None``, only updates tensor meta, chunk id encoder and tile encoder."""
        if chunk:
            local_idx = self.translate_to_local_index(global_idx, row)
            chunk.pop(local_idx)
            self.chunk_id_encoder._encoded[row:, LAST_SEEN_INDEX_COLUMN] -= 1
            self.chunk_id_encoder.is_dirty = True
        self.tensor_meta.pop(global_idx)
        del self.tile_encoder[global_idx]

    def pop_samples(
        self,
        chunk_id: int,
        row: int,
        idxs: List[int],
        is_tile: bool,
    ):
        if not idxs:
            return

        enc = self.chunk_id_encoder

        if is_tile:
            assert len(idxs) == 1, "Tile chunks should only have one sample"
            delete = True
            chunk_ids, _, _ = enc.pop(idxs[0])
        else:
            prev = -1 if row == 0 else enc.array[row - 1][LAST_SEEN_INDEX_COLUMN]
            num_samples_in_chunk = (
                enc.array[row][LAST_SEEN_INDEX_COLUMN] - prev
            ).item()
            num_samples_indexed = len(idxs)

            assert num_samples_indexed <= num_samples_in_chunk

            if num_samples_in_chunk == num_samples_indexed:
                delete = True
            else:
                delete = False

            chunk_ids = [chunk_id]

        chunk_to_update = (
            self.get_chunk_from_chunk_id(chunk_ids[0], copy=True)
            if not delete
            else None
        )
        for idx in idxs:
            self._pop_from_chunk(chunk_to_update, row, idx)

        if delete:
            # tile rows already deleted
            if not is_tile:
                enc._delete_rows([row])
            for chunk_id in chunk_ids:
                chunk_name = ChunkIdEncoder.name_from_id(chunk_id)
                commit_id, tkey = self.get_chunk_commit(chunk_name)
                if commit_id == self.commit_id:
                    chunk_key = get_chunk_key(tkey, chunk_name, commit_id)
                    self.check_remove_active_chunks(chunk_key)
                    try:
                        del self.cache[chunk_key]
                    except KeyError:
                        pass
        else:
            assert chunk_to_update is not None
            self._check_rechunk(chunk_to_update, row)
            if (
                self.active_updated_chunk is not None
                and self.active_updated_chunk.key != chunk_to_update.key  # type: ignore
            ):
                self.write_chunk_to_storage(self.active_updated_chunk)
            self.active_updated_chunk = chunk_to_update

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
    def is_video(self):
        return (
            self.compression in VIDEO_COMPRESSIONS or self.tensor_meta.htype == "video"
        )

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
    def pad_encoder_exists(self) -> bool:
        commit_id = self.commit_id
        if self._pad_encoder is not None and self._pad_encoder_commit_id == commit_id:
            return True
        try:
            key = get_pad_encoder_key(self.key, commit_id)
            self.meta_cache[key]
            return True
        except KeyError:
            return False

    @property
    def _sequence_length(self):
        if self.is_sequence:
            return self.sequence_encoder.num_samples
        return

    @property
    def sequence_encoder(self) -> Optional[SequenceEncoder]:
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
                enc = self.meta_cache.get_deeplake_object(key, SequenceEncoder)
            self._sequence_encoder = enc
            self._sequence_encoder_commit_id = commit_id
            self.meta_cache.register_deeplake_object(key, enc)
        return self._sequence_encoder

    @property
    def pad_encoder(self) -> PadEncoder:
        commit_id = self.commit_id
        if self._pad_encoder is None or self._pad_encoder_commit_id != commit_id:
            commit_id = self.commit_id
            key = get_pad_encoder_key(self.key, commit_id)
            if not self.pad_encoder_exists:
                enc = PadEncoder()
                try:
                    self.meta_cache[key] = enc
                except ReadOnlyModeError:
                    pass
            else:
                enc = self.meta_cache.get_deeplake_object(key, PadEncoder)
            self._pad_encoder = enc
            self._pad_encoder_commit_id = commit_id
            self.meta_cache.register_deeplake_object(key, enc)
        return self._pad_encoder

    def _sequence_numpy(
        self,
        index: Index,
        aslist: bool = False,
        use_data_cache: bool = True,
        fetch_chunks: bool = False,
        pad_tensor: bool = False,
    ):
        arr = self._numpy(
            self._get_flat_index_from_sequence_index(index),
            aslist=aslist,
            use_data_cache=use_data_cache,
            fetch_chunks=fetch_chunks,
            pad_tensor=pad_tensor,
        )
        if self.num_samples == 0:
            return arr
        if isinstance(arr, np.ndarray) and arr.size == 0:
            return self.get_empty_sample()
        if index.subscriptable_at(0) and index.subscriptable_at(1):
            item_lengths = []
            assert self.sequence_encoder is not None
            for i in index.values[0].indices(self._sequence_length):
                item_length = index.length_at(
                    1, -int(np.subtract(*self.sequence_encoder[i]))
                )
                item_lengths.append(item_length)

            if aslist:
                ret = []
                for item_length in item_lengths:
                    ret.append(arr[:item_length])
                    arr = arr[item_length:]
                return ret
            else:
                if len(set(item_lengths)) > 1:
                    raise DynamicTensorNumpyError(self.name, index, "shape")
                try:
                    return arr.reshape(  # type: ignore
                        index.length_at(0, self._sequence_length), -1, *arr.shape[1:]  # type: ignore
                    )
                except ValueError as ve:
                    raise DynamicTensorNumpyError(self.name, index, "shape") from ve
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

        assert self.sequence_encoder is not None
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
        if self.tensor_meta.htype == "class_label":
            samples = self._convert_class_labels(samples)
        if flat_verified_samples:
            verified_samples = []
            for sample in samples:  # type: ignore
                verified_sample = []
                if isinstance(sample, Iterable):
                    for _ in sample:  # type: ignore
                        verified_sample.append(flat_verified_samples[i])
                        i += 1
                    verified_samples.append(verified_sample)
                else:
                    verified_samples.append(flat_verified_samples[i])
                    i += 1

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

    @property
    def _sequence_item_length_range(self):
        """Returns minimum and maximum length of items in a sequence"""
        enc = self.sequence_encoder
        nrows = len(enc._encoded)
        if nrows == 0:
            return 0, 0
        min_ = max_ = enc[0][1] - enc[0][0]
        # sequence length is number of samples in tensor
        for i in range(1, self._sequence_length):
            length = enc[i][1] - enc[i][0]
            if length < min_:
                min_ = length
            elif length > max_:
                max_ = length
        return min_, max_

    def check_link_ready(self):
        return

    def _get_sample_shape_from_provider(
        self, sample_shape_provider, idx, sample_index, flatten
    ):
        try:
            shape = sample_shape_provider(idx)  # type: ignore
        except (
            IndexError
        ):  # Happens during transforms, sample shape tensor is not populated yet
            shape = self.read_shape_for_sample(idx)  # type: ignore

        if isinstance(shape, tuple) and shape == ():
            shape = (0,)
        if self.is_sequence and not flatten:
            shape = self._merge_seq_shape(shape, sample_index)
        return shape

    def _merge_seq_shape(self, shape, sample_index):
        """Merges shapes of sequence items into one shape"""
        if sample_index and not sample_index[0].subscriptable():
            shape = (1, *tuple(shape[sample_index[0].value].tolist()))  # type: ignore
        else:
            is_same = np.all(shape == shape[0, :], axis=0)  # type: ignore
            shape = (len(shape),) + (
                tuple(
                    (
                        int(shape[0, i])  # type: ignore
                        if is_same[i]  # type: ignore
                        else -1
                    )
                    for i in range(shape.shape[1])  # type: ignore
                )
                or (1,)
            )
        return shape

    def _populate_sample_shapes(
        self,
        sample_shapes: np.ndarray,
        index: Index,
        sample_shape_provider: Optional[Callable] = None,
        flatten: bool = False,
    ):
        index_0, sample_index = index.values[0], index.values[1:]
        sample_indices = list(
            index_0.indices(self._sequence_length or self.num_samples)
        )
        num_samples = len(sample_indices)

        sample_ndim = self.ndim() - 1

        bad_shapes = []  # type: ignore
        offset = 0
        for i, idx in enumerate(sample_indices):
            if self.tensor_meta.htype in ("text", "json"):
                shape = (1,)
            elif sample_shape_provider:
                shape = self._get_sample_shape_from_provider(
                    sample_shape_provider, idx, sample_index, flatten
                )
            else:
                self.check_link_ready()
                shape = self.read_shape_for_sample(idx)  # type: ignore
                # if link verification was not done
                if len(shape) > sample_ndim:
                    sample_ndim = len(shape)
                    sample_shapes = np.zeros((num_samples, sample_ndim), dtype=np.int32)

            if flatten:
                assert self.sequence_encoder is not None
                # fill sample shapes with sequence item shapes, no nesting
                start, end = self.sequence_encoder[idx]
                length = end - start
                sample_shapes[offset : offset + length] = shape
                offset += length
            else:
                try:
                    sample_shapes[i] = shape
                except ValueError:
                    # Backwards compatibility for old datasets with
                    # grayscale images stored as (H, W) instead of (H, W, 1)
                    if len(shape) == 2 and sample_shapes.shape[1] == 3:
                        sample_shapes[i] = shape + (1,)
                        bad_shapes.append(i)
        return sample_shapes, bad_shapes

    def _get_total_samples_and_sample_ndim(self, index_0):
        """Returns total number of samples (including sequence items) and sample ndim using first index"""
        tensor_ndim = self.ndim()
        if self.is_sequence:
            sample_indices = list(index_0.indices(self._sequence_length))
            num_samples = sum(
                map(
                    lambda x: x[1] - x[0],
                    [self.sequence_encoder[i] for i in sample_indices],
                )
            )
            sample_ndim = tensor_ndim - 2
        else:
            num_samples = index_0.length(self.num_samples)
            sample_ndim = tensor_ndim - 1
        return num_samples, sample_ndim

    def _group_flat_shapes(self, sample_shapes, index_0, sample_ndim):
        """Groups shapes of flattened sequence items"""
        sample_indices = list(index_0.indices(self._sequence_length))
        num_samples = len(sample_indices)
        seq_item_length = self.sequence_encoder[sample_indices[0]]
        seq_item_length = seq_item_length[1] - seq_item_length[0]
        # try reshape to (num_samples, seq_item_length, sample_ndim)
        try:
            if isinstance(sample_shapes, list):
                raise ValueError
            sample_shapes = sample_shapes[np.newaxis, :].reshape(
                num_samples, seq_item_length, sample_ndim
            )
            return sample_shapes
        except ValueError:
            sample_shapes_list = []
            offset = 0
            for i, idx in enumerate(sample_indices):
                start, end = self.sequence_encoder[idx]
                length = end - start
                sample_shapes_list.append(sample_shapes[offset : offset + length])
                offset += length
            return sample_shapes_list

    def shapes(
        self,
        index: Index,
        sample_shape_provider: Optional[Callable] = None,
        pad_tensor: bool = False,
        convert_bad_to_list: bool = True,
    ):
        if len(index) > 1:
            raise IndexError("`.shapes` only accepts indexing on the primary axis.")

        index_0 = index.values[0]
        num_samples, sample_ndim = self._get_total_samples_and_sample_ndim(index_0)

        sample_shapes = np.zeros((num_samples, sample_ndim), dtype=np.int32)

        if (
            index.is_trivial()
            or self.tensor_meta.min_shape == self.tensor_meta.max_shape
            or num_samples == 0
        ):
            shape = self.shape_interval(index).astuple()[1:]
        else:
            shape = None

        if (
            not index_0.subscriptable()
            and pad_tensor
            and index_0.value >= self.tensor_length  # type: ignore
        ):
            shape = self.get_empty_sample().shape

        if shape is None or None in shape or self.tensor_meta.is_link:
            sample_shapes, bad_shapes = self._populate_sample_shapes(
                sample_shapes,
                index,
                sample_shape_provider,
                flatten=True if self.is_sequence else False,
            )
            # convert to list if grayscale images were stored as (H, W) instead of (H, W, 1)
            if bad_shapes and convert_bad_to_list:
                sample_shapes = sample_shapes.tolist()
                for i in bad_shapes:
                    sample_shapes[i] = sample_shapes[i][:-1]
            if self.is_sequence:
                sample_shapes = self._group_flat_shapes(
                    sample_shapes, index_0, sample_ndim
                )
        else:
            sample_shapes[:] = shape

        return sample_shapes

    def _apply_deeper_indexing(self, sample_shapes, num_samples, sample_index):
        """Applies rest of the indexing to the sample shapes. Inplace operation."""
        squeeze_dims = set()
        for i in range(num_samples):
            for j in range(len(sample_index)):
                if sample_index[j].subscriptable():
                    if sample_shapes[i, j] != -1:
                        sample_shapes[i, j] = sample_index[j].length(
                            sample_shapes[i, j]
                        )
                else:
                    squeeze_dims.add(j)
        return squeeze_dims

    def _sample_shapes_to_shape(self, sample_shapes, squeeze_dims, sample_ndim):
        is_same = np.all(sample_shapes == sample_shapes[0, :], axis=0)
        shape = [  # type: ignore
            (
                int(sample_shapes[0, i])
                if sample_shapes[0, i] != -1 and is_same[i]
                else None
            )
            for i in range(sample_ndim)
        ]

        return tuple(shape[i] for i in range(len(shape)) if i not in squeeze_dims)

    def shape(
        self,
        index: Index,
        sample_shape_provider: Optional[Callable] = None,
        pad_tensor: bool = False,
    ) -> Tuple[Optional[int], ...]:
        tensor_ndim = self.ndim()

        if len(index) > tensor_ndim:
            raise IndexError(
                f"Too many indices for tensor. Tensor is rank {tensor_ndim} but {len(index)} indices were provided."
            )

        index_0, sample_index = index.values[0], index.values[1:]
        if (
            not index_0.subscriptable()
            and pad_tensor
            and index_0.value >= self.tensor_length  # type: ignore
        ):
            return self.get_empty_sample().shape

        num_samples = index_0.length(self._sequence_length or self.num_samples)
        if self.tensor_meta.min_shape == self.tensor_meta.max_shape:
            if index_0.is_trivial() or num_samples == 0:
                shape = self.shape_interval(index).astuple()
                return shape
            else:
                shape = self.shape_interval(index).astuple()[1:]
        else:
            shape = None

        sample_ndim = tensor_ndim - 1
        sample_shapes = np.zeros((num_samples, sample_ndim), dtype=np.int32)

        if shape is None or None in shape or self.tensor_meta.is_link:
            sample_shapes, bad_shapes = self._populate_sample_shapes(
                sample_shapes, index, sample_shape_provider, flatten=False
            )
            sample_ndim = sample_shapes.shape[1]
        else:
            sample_shapes[:] = shape

        squeeze_dims = self._apply_deeper_indexing(
            sample_shapes, num_samples, sample_index
        )
        shape = self._sample_shapes_to_shape(sample_shapes, squeeze_dims, sample_ndim)

        if index_0.subscriptable():
            shape = (num_samples, *shape)  # type: ignore

        return shape  # type: ignore

    def ndim(self, index: Optional[Index] = None) -> int:
        ndim = len(self.tensor_meta.min_shape) + 1
        if self.is_sequence:
            ndim += 1
        if index:
            for idx in index.values:
                if not idx.subscriptable():
                    ndim -= 1
        return ndim

    def shape_interval(
        self, index: Index, sample_shape_provider: Optional[Callable] = None
    ) -> ShapeInterval:
        """Returns a `ShapeInterval` object that describes this tensor's shape more accurately. Length is included.

        Args:
            index (Index): Index to use for shape calculation.
            sample_shape_provider (Optional, Callable): Function that returns a sample shape for a given index.

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
            tensor_length = index.length(self._sequence_length)
        else:
            tensor_length = index.length(meta.length)

        if index.is_trivial() or meta.min_shape == meta.max_shape or tensor_length == 0:
            if self.is_sequence:
                min_item_length, max_item_length = self._sequence_item_length_range
                min_length = [tensor_length, min_item_length]
                max_length = [tensor_length, max_item_length]
            else:
                min_length = max_length = [tensor_length]
            min_shape = min_length + list(meta.min_shape)
            max_shape = max_length + list(meta.max_shape)
        else:
            # need to fetch all shapes for the index
            shapes = self.shapes(
                index, sample_shape_provider, convert_bad_to_list=False
            )
            if self.is_sequence:
                if isinstance(shapes, np.ndarray):
                    # uniform sequence of shape (num_samples, num_items, ...)
                    min_shape = [*shapes.shape[:-1], *np.amin(shapes, axis=(0, 1))]
                    max_shape = [*shapes.shape[:-1], *np.amax(shapes, axis=(0, 1))]
                else:
                    # non-uniform sequence
                    item_lengths = list(map(len, shapes))
                    min_item_length, max_item_length = min(item_lengths), max(
                        item_lengths
                    )
                    min_item_shape = np.amin(
                        list(map(lambda x: np.amin(x, axis=0), shapes)), axis=0
                    )
                    max_item_shape = np.amax(
                        list(map(lambda x: np.amax(x, axis=0), shapes)), axis=0
                    )
                    min_shape = [len(shapes), min_item_length, *min_item_shape]
                    max_shape = [len(shapes), max_item_length, *max_item_shape]
            else:
                min_shape = [len(shapes), *np.amin(shapes, axis=0)]
                max_shape = [len(shapes), *np.amax(shapes, axis=0)]

        return ShapeInterval(min_shape, max_shape)

    def _transform_callback(
        self, samples, flat: Optional[bool], progressbar: bool = False
    ):
        """Used in transforms to handle linked tensors."""
        updated_tensors = {}
        try:
            for k, v in self.tensor_meta.links.items():
                if self._all_chunk_engines and (
                    flat is None or v["flatten_sequence"] == flat
                ):
                    tensor = self.version_state["full_tensors"][k]
                    func = get_link_transform(v["extend"])
                    meta = self.tensor_meta
                    vs = func(
                        samples,
                        factor=(
                            tensor.info.downsampling_factor
                            if func == extend_downsample
                            else None
                        ),
                        compression=meta.sample_compression,
                        htype=meta.htype,
                        link_creds=self.link_creds,
                        progressbar=progressbar,
                        tensor_meta=self.tensor_meta,
                    )
                    dtype = tensor.dtype
                    if dtype:
                        if isinstance(vs, np.ndarray):
                            vs = cast_to_type(vs, dtype)
                        else:
                            vs = [cast_to_type(v, dtype) for v in vs]
                    chunk_engine = self._all_chunk_engines[k]
                    updated_tensors[k] = chunk_engine.tensor_length
                    chunk_engine.extend(vs)
                    chunk_engine._transform_callback(vs, flat)
        except Exception:
            for k, num_samples in updated_tensors.items():
                assert self._all_chunk_engines is not None
                chunk_engine = self._all_chunk_engines[k]
                chunk_engine.pop(list(range(num_samples, chunk_engine.tensor_length)))
            raise

    def _transform_pop_callback(self, index: int):
        if self._all_chunk_engines:
            if self.is_sequence:
                flat_links: List[str] = []
                links: List[str] = []
                for link, props in self.tensor_meta.links.items():
                    (flat_links if props["flatten_sequence"] else links).append(link)

                if flat_links:
                    seq_enc = self.sequence_encoder
                    assert seq_enc is not None
                    assert self._all_chunk_engines is not None
                    for link in flat_links:
                        link_chunk_engine = self._all_chunk_engines[link]
                        link_chunk_engine.pop(list(range(*seq_enc[index])))
            else:
                links = list(self.tensor_meta.links.keys())
            [self._all_chunk_engines[link].pop() for link in links]

    def get_empty_sample(self, index: Optional[Index] = None):
        if self.num_samples == 0:
            raise ValueError("This tensor has no samples, cannot get empty sample.")
        htype = self.tensor_meta.htype
        dtype = self.tensor_meta.dtype
        if htype in ("text", "json", "list", "tag"):
            sample = get_empty_text_like_sample(htype)
        else:
            ndim = len(self.tensor_meta.max_shape)
            if self.is_sequence:
                ndim += 1
            shape = (0,) * ndim
            sample = np.ones(shape, dtype=dtype)

        if index:
            try:
                return sample[tuple(entry.value for entry in index.values[1:])]
            except IndexError:
                pass

        return sample

    @property
    def is_text_like(self):
        return (
            self.tensor_meta.htype in {"text", "json", "list", "tag"}
            or self.tensor_meta.is_link
        )

    def check_remove_active_chunks(self, chunk_key):
        if (
            self.active_appended_chunk is not None
            and self.active_appended_chunk.key == chunk_key
        ):
            self.active_appended_chunk = None
        if (
            self.active_updated_chunk is not None
            and self.active_updated_chunk.key == chunk_key
        ):
            self.active_updated_chunk = None

    def get_avg_chunk_size(self):
        num_chunks, num_samples = self.num_chunks, self.num_samples
        max_shape = self.tensor_meta.max_shape
        dtype = self.tensor_meta.dtype
        if dtype in ("Any", "List", None):
            return None
        shape = [num_samples] + max_shape
        nbytes = 1
        for dim in shape:  # not using np.prod to avoid overflow
            nbytes *= dim
        nbytes = nbytes * np.dtype(dtype).itemsize
        avg_chunk_size = nbytes / num_chunks
        return avg_chunk_size

    def __getstate__(self):
        state = self.__dict__

        # remove cached chunks
        state["_active_appended_chunk"] = None
        state["_active_updated_chunk"] = None

        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
