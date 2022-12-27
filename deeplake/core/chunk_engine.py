from collections import OrderedDict
from deeplake.client.log import logger
import deeplake
import numpy as np
from tqdm import tqdm  # type: ignore
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
from deeplake.core.linked_sample import LinkedSample
from deeplake.core.meta.encode.base_encoder import LAST_SEEN_INDEX_COLUMN
from deeplake.core.serialize import HEADER_SIZE_BYTES
from deeplake.core.tensor_link import get_link_transform
from deeplake.core.version_control.commit_diff import CommitDiff
from deeplake.core.partial_reader import PartialReader
from deeplake.core.version_control.commit_node import CommitNode  # type: ignore
from deeplake.core.version_control.commit_chunk_set import CommitChunkSet  # type: ignore
from typing import Any, Dict, List, Optional, Sequence, Union, Callable
from deeplake.core.meta.encode.tile import TileEncoder
from deeplake.core.storage.provider import StorageProvider
from deeplake.core.storage import S3Provider, GCSProvider
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
from deeplake.core.meta.tensor_meta import TensorMeta
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
    get_tensor_commit_diff_key,
    get_tensor_meta_key,
    get_chunk_key,
    get_tensor_commit_chunk_set_key,
    get_tensor_meta_key,
    get_tensor_tile_encoder_key,
    get_tensor_info_key,
)
from deeplake.util.exceptions import (
    CorruptedMetaError,
    DynamicTensorNumpyError,
    ReadOnlyModeError,
    SampleHtypeMismatchError,
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
        self._is_temp_label_tensor: bool = False
        self._hash_label_map: Dict[int, str] = OrderedDict()
        self._sample_compression = None
        self._chunk_compression = None

        tensor_meta = self.tensor_meta
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

        self.cached_data: Optional[np.ndarray] = None
        self.cache_range: range = range(0)

        self._chunk_args = None
        self._num_samples_per_chunk: Optional[int] = None
        self.write_initialization_done = False
        self.start_chunk = None

    @property
    def sample_compression(self):
        return self._sample_compression

    @property
    def chunk_compression(self):
        return self._chunk_compression

    @property
    def is_data_cachable(self):
        tensor_meta = self.tensor_meta
        return (
            self.chunk_class == UncompressedChunk
            and tensor_meta.htype not in ["text", "json", "list", "polygon"]
            and tensor_meta.max_shape
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
                cset = self.meta_cache.get_deeplake_object(key, CommitChunkSet)
            self._commit_chunk_set = cset
            self._commit_chunk_set_commit_id = commit_id
            self.meta_cache.register_deeplake_object(key, cset)
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
        """Returns the length of the primary axis of the tensor.
        Ignores any applied indexing and returns the total length.
        """
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
        chunk_name = ChunkIdEncoder.name_from_id(chunk_id)
        chunk_commit_id = self.get_chunk_commit(chunk_name)
        chunk_key = get_chunk_key(self.key, chunk_name, chunk_commit_id)
        chunk = self.get_chunk(chunk_key, partial_chunk_bytes=partial_chunk_bytes)
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
                chunk = self.cache.get_deeplake_object(
                    chunk_key, self.chunk_class, meta=self.chunk_args, url=True
                )
        if not stream:
            chunk = self.cache.get_deeplake_object(
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
                    chunk_set = self.meta_cache.get_deeplake_object(
                        chunk_set_key, CommitChunkSet
                    ).chunks
            except Exception:
                commit_chunk_set = CommitChunkSet()
                try:
                    self.meta_cache[chunk_set_key] = commit_chunk_set
                except ReadOnlyModeError:
                    # put CommitChunkSet in deeplake_objects to keep in cache temporarily, but won't write to storage
                    # this shouldn't happen in latest version of deeplake, chunk set would always be present
                    self.meta_cache.deeplake_objects[chunk_set_key] = commit_chunk_set
                chunk_set = set()
            if chunk_name in chunk_set:
                return commit_id
            cur_node = cur_node.parent  # type: ignore
        # the first commit doesn't have a commit chunk set, so any chunk that wasn't found belongs to the first commit
        return FIRST_COMMIT_ID

    def _write_initialization(self):
        ffw_chunk_id_encoder(self.chunk_id_encoder)

    def _convert_to_list(self, samples):
        return False

    def check_each_sample(self, samples, verify=True):
        return

    def _sanitize_samples(self, samples, verify=True):
        check_samples_type(samples)
        if isinstance(samples, list):
            samples = [None if is_empty_list(sample) else sample for sample in samples]
        verified_samples = self.check_each_sample(samples, verify=verify)
        tensor_meta = self.tensor_meta
        all_empty = all(sample is None for sample in samples)
        if tensor_meta.htype is None and not all_empty:
            tensor_meta.set_htype(get_htype(samples))
        if tensor_meta.dtype is None and not all_empty:
            tensor_meta.set_dtype(get_dtype(samples[0]))
        if self._convert_to_list(samples):
            samples = list(samples)
        if self._is_temp_label_tensor:
            samples = verified_samples = convert_to_hash(samples, self._hash_label_map)
        elif tensor_meta.htype in ("image.gray", "image.rgb"):
            mode = "L" if tensor_meta.htype == "image.gray" else "RGB"
            converted = []
            for sample in samples:
                if isinstance(sample, Sample):
                    converted.append(convert_sample(sample, mode))
                elif isinstance(sample, np.ndarray):
                    converted.append(convert_img_arr(sample, mode))
                else:
                    raise SampleHtypeMismatchError(tensor_meta.htype, type(sample))
            samples = verified_samples = converted
        elif tensor_meta.htype == "class_label":
            samples = verified_samples = self._convert_class_labels(samples)
        elif tensor_meta.htype == "polygon":
            samples = verified_samples = [
                p if isinstance(p, Polygons) else Polygons(p, dtype=tensor_meta.dtype)
                for p in samples
            ]
        return samples, verified_samples

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

        Returns:
            Tuple[List[BaseChunk], Dict[Any, Any]]
        """
        extending = start_chunk_row is None and register
        lengths = None
        num_samples = self.num_samples
        orig_meta_length = self.tensor_meta.length
        incoming_num_samples = len(samples)
        if extending:
            num_samples = len(samples)
            enc_ids = []
            enc_count = [0]
            if self.tensor_meta.htype == "text" and (
                self.chunk_class != SampleCompressedChunk
            ):
                lengths = np.zeros(len(samples), dtype=np.uint32)
                for i, s in enumerate(samples):
                    try:
                        lengths[i] = s.__len__()
                    except AttributeError:  # None
                        lengths[i] = 0
                    except TypeError:  # Numpy scalar str
                        lengths[i] = str(s).__len__()
        extra_args = {"lengths": lengths}
        current_chunk = start_chunk
        updated_chunks = []
        if current_chunk is None:
            current_chunk = self._create_new_chunk(
                register and start_chunk_row is not None
            )
            current_chunk._update_tensor_meta_length = False
            updated_chunks.append(current_chunk)
            if extending:
                enc_ids.append(current_chunk.id)  # type: ignore
        else:
            current_chunk._update_tensor_meta_length = False
            if extending:
                enc_ids.append(None)
        enc = self.chunk_id_encoder
        tiles = {}
        if register and update_commit_diff:
            commit_diff = self.commit_diff
        if progressbar:
            pbar = tqdm(total=len(samples))
        if not isinstance(samples, list) and not (
            isinstance(samples, np.ndarray) and self._numpy_extend_optimization_enabled
        ):
            # Note: in the future we can get rid of this conversion of sample compressed chunks too by predicting the compression ratio.
            samples = list(samples)
        while len(samples) > 0:
            num_samples_added = current_chunk.extend_if_has_space(
                samples, update_tensor_meta=update_tensor_meta, **extra_args  # type: ignore
            )  # type: ignore
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
                    enc_ids.append(current_chunk.id)  # type: ignore
                    enc_count.append(0)
                updated_chunks.append(current_chunk)
            elif num_samples_added == PARTIAL_NUM_SAMPLES:
                sample = samples[0]
                if sample.is_first_write:
                    if register:
                        if start_chunk_row is not None:
                            enc.register_samples(1)
                        else:
                            enc_count[-1] += 1
                if sample.is_last_write:
                    tiles[
                        incoming_num_samples
                        - len(samples)
                        + bool(register) * orig_meta_length
                    ] = (
                        sample.sample_shape,
                        sample.tile_shape,
                    )
                    samples = samples[1:]
                    if lengths is not None:
                        lengths = lengths[1:]
                    num_samples_added = 1
                    num_samples += 1
                else:
                    num_samples_added = 0
                if len(samples) > 0:
                    current_chunk = self._create_new_chunk(
                        register and start_chunk_row is not None, row=start_chunk_row
                    )
                    current_chunk._update_tensor_meta_length = False
                    if start_chunk_row is not None:
                        start_chunk_row += 1
                    elif register:
                        enc_ids.append(current_chunk.id)  # type: ignore
                        enc_count.append(0)
                    updated_chunks.append(current_chunk)
            elif num_samples_added == FAST_EXTEND_BAIL:
                num_samples_added = 0
                samples = list(samples)
            else:
                if not updated_chunks:
                    updated_chunks.append(current_chunk)
                num = int(num_samples_added)
                if register:
                    if start_chunk_row is not None:
                        enc.register_samples(num, row=start_chunk_row)
                    else:
                        enc_count[-1] += num
                samples = samples[num:]
                if lengths is not None:
                    lengths = lengths[num:]
                num_samples += num
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

        if register:
            return updated_chunks
        return updated_chunks, tiles

    def register_new_creds(self, num_samples_added, samples):
        return

    def update_creds(self, sample_index, sample):
        return

    def _extend(self, samples, progressbar, pg_callback=None, update_commit_diff=True):
        if isinstance(samples, deeplake.Tensor):
            samples = tqdm(samples) if progressbar else samples
            for sample in samples:
                self._extend(
                    [sample],
                    update_commit_diff=update_commit_diff,
                    progressbar=False,
                    pg_callback=pg_callback,
                )  # TODO optimize this
            return
        if len(samples) == 0:
            return
        samples, verified_samples = self._sanitize_samples(samples)
        self._samples_to_chunks(
            samples,
            start_chunk=self.last_appended_chunk(),
            register=True,
            progressbar=progressbar,
            update_commit_diff=update_commit_diff,
            pg_callback=pg_callback,
        )
        return verified_samples

    def extend(
        self,
        samples,
        progressbar: bool = False,
        link_callback: Optional[Callable] = None,
        pg_callback=None,
    ):
        assert not (progressbar and pg_callback)
        self.check_link_ready()
        if not self.write_initialization_done:
            self._write_initialization()
            self.write_initialization_done = True

        initial_autoflush = self.cache.autoflush
        self.cache.autoflush = False

        if self.is_sequence:
            samples = tqdm(samples) if progressbar else samples
            verified_samples = []
            for sample in samples:
                if sample is None:
                    sample = []
                verified_sample = self._extend(
                    sample, progressbar=False, update_commit_diff=False
                )
                self.sequence_encoder.register_samples(len(sample), 1)
                self.commit_diff.add_data(1)
                verified_samples.append(verified_sample or sample)
            if link_callback:
                samples = [None if is_empty_list(s) else s for s in verified_samples]
                link_callback(verified_samples, flat=False)
                for s in verified_samples:
                    link_callback(s, flat=True)

        else:
            verified_samples = (
                self._extend(samples, progressbar, pg_callback=pg_callback) or samples
            )
            if link_callback:
                if not isinstance(verified_samples, np.ndarray):
                    samples = [
                        None if is_empty_list(s) else s for s in verified_samples
                    ]
                link_callback(samples, flat=None)

        self.cache.autoflush = initial_autoflush
        self.cache.maybe_flush()

    def _create_new_chunk(self, register=True, row: Optional[int] = None) -> BaseChunk:
        """Creates and returns a new `Chunk`. Automatically creates an ID for it and puts a reference in the cache."""
        chunk_id = self.chunk_id_encoder.generate_chunk_id(register=register, row=row)
        chunk = self.chunk_class(*self.chunk_args)  # type: ignore
        chunk_name = ChunkIdEncoder.name_from_id(chunk_id)  # type: ignore
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
        if num_samples_to_pad > 0:
            if self.num_samples == 0:
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

        self.extend([value], link_callback=extend_link_callback)

    def update(
        self,
        index: Index,
        samples: Union[np.ndarray, Sequence[InputSample], InputSample],
        operator: Optional[str] = None,
        link_callback: Optional[Callable] = None,
    ):
        """Update data at `index` with `samples`."""
        self.check_link_ready()
        (self._sequence_update if self.is_sequence else self._update)(  # type: ignore
            index,
            samples,
            operator,
            link_callback=link_callback,
        )

    def _get_samples_to_move(self, chunk) -> List[Sample]:
        decompress = isinstance(chunk, ChunkCompressedChunk)
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

    def _get_chunk_samples(self, chunk) -> List[Sample]:
        decompress = isinstance(chunk, ChunkCompressedChunk)
        all_samples_in_chunk: List[Sample] = []

        for idx in range(chunk.num_samples):
            sample_data = chunk.read_sample(idx, decompress=decompress)
            sample_shape = chunk.shapes_encoder[idx]
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
        if decompress:
            sample = Sample(array=sample_data, shape=sample_shape)
        else:
            sample = Sample(
                buffer=sample_data,
                shape=sample_shape,
                compression=compression,
                dtype=dtype,
            )

        if self.tensor_meta.htype in ("json", "text", "list"):
            sample.htype = self.tensor_meta.htype
        if self.tensor_meta.is_link:
            sample.htype = "text"
            sample = LinkedSample(sample.array[0])
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
        samples, _ = self._sanitize_samples(samples_to_move, verify=False)
        self._samples_to_chunks(
            samples,
            start_chunk=new_chunk,
            register=True,
            update_commit_diff=True,
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
        samples, _ = self._sanitize_samples(samples_to_move, verify=False)
        to_chunk.is_dirty = True
        self.active_updated_chunk = to_chunk
        self._samples_to_chunks(
            samples,
            start_chunk=to_chunk,
            register=True,
            update_commit_diff=True,
            update_tensor_meta=False,
            start_chunk_row=to_chunk_row,
            register_creds=False,
        )
        self.chunk_id_encoder.delete_chunk_id(row=from_chunk_row)
        try:
            del self.cache[from_chunk.key]  # type: ignore
        except KeyError:
            pass
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
        next_chunk_commit_id = self.get_chunk_commit(next_chunk_name)
        chunk_key = get_chunk_key(self.key, next_chunk_name, next_chunk_commit_id)
        next_chunk_size = self.cache.get_object_size(chunk_key)
        next_chunk = self.get_chunk_from_chunk_id(int(next_chunk_id))
        if next_chunk_size + chunk.num_data_bytes < next_chunk.min_chunk_size:
            # merge with next chunk
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
        prev_chunk_commit_id = self.get_chunk_commit(prev_chunk_name)
        prev_chunk_key = get_chunk_key(self.key, prev_chunk_name, prev_chunk_commit_id)
        prev_chunk_size = self.cache.get_object_size(prev_chunk_key)
        prev_chunk = self.get_chunk_from_chunk_id(int(prev_chunk_id))
        if prev_chunk_size + chunk.num_data_bytes < prev_chunk.min_chunk_size:
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
        if self.tensor_meta.name.startswith("_"):
            return (
                self.tensor_meta.name.endswith("_shape")
                or self.tensor_meta.name.endswith("_id")
                or self.tensor_meta.name.endswith("_info")
            )
        return False

    def _check_rechunk(self, chunk: BaseChunk, chunk_row: int):
        """function to check if there is a need to re-chunk the current one"""

        if self.tensor_meta.name and self.is_tensor_hidden():
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

        if operator is not None:
            return self._update_with_operator(index, samples, operator)

        enc = self.chunk_id_encoder
        index_length = index.length(self.num_samples)
        samples = make_sequence(samples, index_length)
        verified_samples = self.check_each_sample(samples)
        if self.tensor_meta.htype == "class_label":
            samples = self._convert_class_labels(samples)
        nbytes_after_updates = []
        global_sample_indices = tuple(index.values[0].indices(self.num_samples))
        is_sequence = self.is_sequence
        for i, sample in enumerate(samples):  # type: ignore
            sample = None if is_empty_list(sample) else sample
            global_sample_index = global_sample_indices[i]  # TODO!
            if self._is_tiled_sample(global_sample_index):
                self._update_tiled_sample(global_sample_index, index, sample)
            else:
                chunk = self.get_chunks_for_sample(global_sample_index, copy=True)[0]
                local_sample_index = enc.translate_index_relative_to_chunks(
                    global_sample_index
                )

                if len(index.values) <= 1 + int(self.is_sequence):
                    chunk.update_sample(local_sample_index, sample)
                else:
                    orig_sample = chunk.read_sample(local_sample_index, copy=True)
                    orig_sample[tuple(e.value for e in index.values[1:])] = sample
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
                self._check_rechunk(
                    chunk, chunk_row=enc.__getitem__(global_sample_index, True)[0][1]
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
        return tensor_meta.min_shape == tensor_meta.max_shape

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
                stop = self.num_samples + start

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

    def get_video_sample(self, global_sample_index, index, decompress=True):
        enc = self.chunk_id_encoder
        chunk_ids = enc[global_sample_index]
        local_sample_index = enc.translate_index_relative_to_chunks(global_sample_index)
        chunk, stream = self.get_video_chunk(chunk_ids[0])
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

    def get_chunk_info(self, global_sample_index, fetch_chunks):
        """Returns the chunk_id, row and worst case header size of chunk containing the given sample."""
        enc = self.chunk_id_encoder
        out = enc.__getitem__(global_sample_index, return_row_index=True)
        chunk_id, row = out[0][0], out[0][1]

        worst_case_header_size = 0
        num_samples_in_chunk = -1
        if (
            not fetch_chunks
            and self.chunk_class != ChunkCompressedChunk
            and isinstance(self.base_storage, (S3Provider, GCSProvider))
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

    def get_basic_sample(self, global_sample_index, index, fetch_chunks=False):
        enc = self.chunk_id_encoder
        chunk_id, row, worst_case_header_size = self.get_chunk_info(
            global_sample_index, fetch_chunks
        )
        local_sample_index = enc.translate_index_relative_to_chunks(global_sample_index)
        chunk = self.get_chunk_from_chunk_id(
            chunk_id, partial_chunk_bytes=worst_case_header_size
        )
        ret = chunk.read_sample(
            local_sample_index,
            cast=self.tensor_meta.htype != "dicom",
        )
        if len(index) > 1:
            ret = ret[tuple(entry.value for entry in index.values[1:])]
        return ret

    def get_non_tiled_sample(self, global_sample_index, index, fetch_chunks=False):
        if self.is_video:
            return self.get_video_sample(global_sample_index, index)
        return self.get_basic_sample(
            global_sample_index, index, fetch_chunks=fetch_chunks
        )

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
            lambda chunk_id: self.get_chunk_from_chunk_id(chunk_id).read_sample(
                0, is_tile=True
            ),
            otypes=[object],
        )(required_tile_ids)
        sample = coalesce_tiles(tiles, tile_shape, None, self.tensor_meta.dtype)
        sample = sample[sample_index]
        return sample

    def get_single_sample(
        self, global_sample_index, index, fetch_chunks=False, pad_tensor=False
    ):
        if pad_tensor and global_sample_index >= self.tensor_meta.length:
            sample = self.get_empty_sample()
            try:
                return sample[tuple(entry.value for entry in index.values[1:])]
            except IndexError:
                return sample

        if not self._is_tiled_sample(global_sample_index):
            sample = self.get_non_tiled_sample(
                global_sample_index, index, fetch_chunks=fetch_chunks
            )
        elif len(index.values) == 1:
            sample = self.get_full_tiled_sample(global_sample_index)
        else:
            sample = self.get_partial_tiled_sample(global_sample_index, index)

        return sample

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
            for global_sample_index in index.values[0].indices(length):
                sample = self.get_single_sample(
                    global_sample_index,
                    index,
                    fetch_chunks=fetch_chunks,
                    pad_tensor=pad_tensor,
                )
                check_sample_shape(sample.shape, last_shape, self.key, index, aslist)
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
            if pad_tensor and global_sample_index >= self.tensor_meta.length:
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
            return [
                ChunkIdEncoder.name_from_id(chunk_id)
                for chunk_id in self.chunk_id_encoder.array[:, CHUNK_ID_COLUMN]
            ]  # type: ignore
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

    def pop(self, global_sample_index: int):
        self._write_initialization()
        if self.tensor_meta.length == 0:
            raise ValueError("There are no samples to pop")
        if global_sample_index < 0 or global_sample_index >= self.tensor_meta.length:
            raise IndexError(
                f"Index {global_sample_index} is out of range for tensor of length {self.tensor_meta.length}"
            )

        self.cached_data = None
        initial_autoflush = self.cache.autoflush
        self.cache.autoflush = False

        self.commit_diff.pop(global_sample_index)
        if self.is_sequence:
            # pop in reverse order else indices get shifted
            for idx in reversed(range(*self.sequence_encoder[global_sample_index])):
                self.pop_item(idx)
            self.sequence_encoder.pop(global_sample_index)
        else:
            self.pop_item(global_sample_index)

        self.cache.autoflush = initial_autoflush
        self.cache.maybe_flush()

    def pop_item(self, global_sample_index):
        enc = self.chunk_id_encoder
        if not self._is_tiled_sample(global_sample_index):
            local_sample_index = enc.translate_index_relative_to_chunks(
                global_sample_index
            )
        chunk_ids, rows, delete = enc.pop(global_sample_index)
        if len(chunk_ids) > 1:  # Tiled sample, delete all chunks
            pass
        elif not delete:  # There are other samples in the last chunk
            chunk_to_update = self.get_chunk_from_chunk_id(chunk_ids[0], copy=True)
            chunk_to_update.pop(local_sample_index)

            self._check_rechunk(chunk_to_update, chunk_row=rows[0])

            if (
                self.active_updated_chunk is not None
                and self.active_updated_chunk.key != chunk_to_update.key  # type: ignore
            ):
                self.write_chunk_to_storage(self.active_updated_chunk)
            self.active_updated_chunk = chunk_to_update
        if delete:
            for chunk_id in chunk_ids:
                chunk_name = ChunkIdEncoder.name_from_id(chunk_id)
                commit_id = self.get_chunk_commit(chunk_name)
                if commit_id == self.commit_id:
                    chunk_key = get_chunk_key(self.key, chunk_name, commit_id)
                    self.check_remove_active_chunks(chunk_key)
                    try:
                        del self.cache[chunk_key]
                    except KeyError:
                        pass
        del self.tile_encoder[global_sample_index]
        self.tensor_meta.pop(global_sample_index)

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
                enc = self.meta_cache.get_deeplake_object(key, SequenceEncoder)
            self._sequence_encoder = enc
            self._sequence_encoder_commit_id = commit_id
            self.meta_cache.register_deeplake_object(key, enc)
        return self._sequence_encoder

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
        if isinstance(arr, np.ndarray) and arr.size == 0:
            return self.get_empty_sample()
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
        if self.tensor_meta.htype == "class_label":
            samples = self._convert_class_labels(samples)
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

    @property
    def _sequence_item_length_range(self):
        """Returns minimum and maximum length of items in a sequence"""
        enc = self.sequence_encoder
        nrows = len(enc._encoded)
        if nrows == 0:
            return 0, 0
        min_ = max_ = enc[0][1] - enc[0][0]
        for i in range(1, nrows):
            length = enc[i][1] - enc[i][0]
            if length < min_:
                min_ = length
            elif length > max_:
                max_ = length
        return min_, max_

    def check_link_ready(self):
        return

    def shape(
        self, index: Index, sample_shape_provider: Optional[Callable] = None
    ) -> Tuple[Optional[int], ...]:
        shape = self.shape_interval.astuple()
        idxs = index.values
        skip_dims = 0
        if None in shape or self.tensor_meta.is_link:
            if not idxs[0].subscriptable():
                if self.tensor_meta.htype in ("text", "json"):
                    shape = (1,)
                else:
                    if sample_shape_provider:
                        try:
                            shape = sample_shape_provider(idxs[0].value)  # type: ignore
                            if self.is_sequence:
                                if len(idxs) > 1 and not idxs[1].subscriptable():
                                    shape = tuple(shape[idxs[1].value].tolist())  # type: ignore
                                    skip_dims += 1
                                else:
                                    shape = (len(shape),) + (
                                        tuple(
                                            int(shape[0, i])  # type: ignore
                                            if np.all(shape[:, i] == shape[0, i])  # type: ignore
                                            else None
                                            for i in range(shape.shape[1])  # type: ignore
                                        )
                                        or (1,)
                                    )

                        except IndexError:  # Happens during transforms, sample shape tensor is not populated yet
                            shape = self.read_shape_for_sample(idxs[0].value)  # type: ignore
                    else:
                        self.check_link_ready()
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
            seq_length = self._sequence_length
            min_item_length, max_item_length = self._sequence_item_length_range
            min_length = [seq_length, min_item_length]
            max_length = [seq_length, max_item_length]
        else:
            min_length = max_length = [meta.length]
        min_shape = min_length + list(meta.min_shape)
        max_shape = max_length + list(meta.max_shape)

        return ShapeInterval(min_shape, max_shape)

    def _transform_callback(self, samples, flat: Optional[bool]):
        """Used in transforms to handle linked tensors."""
        for k, v in self.tensor_meta.links.items():
            if self._all_chunk_engines and (
                flat is None or v["flatten_sequence"] == flat
            ):
                self._all_chunk_engines[k].extend(
                    get_link_transform(v["extend"])(samples)
                )

    def get_empty_sample(self):
        if self.num_samples == 0:
            raise ValueError("This tensor has no samples, cannot get empty sample.")
        htype = self.tensor_meta.htype
        dtype = self.tensor_meta.dtype
        if htype in ("text", "json", "list"):
            return get_empty_text_like_sample(htype)
        ndim = len(self.tensor_meta.max_shape)
        if self.is_sequence:
            ndim += 1
        shape = (0,) * ndim
        return np.ones(shape, dtype=dtype)

    @property
    def is_text_like(self):
        return (
            self.tensor_meta.htype in {"text", "json", "list"}
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
