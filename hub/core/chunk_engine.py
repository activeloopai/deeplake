import hub
from hub.util.chunks import chunk_name_from_id
from hub.core.tiling.optimize import TileOptimizer
from hub.util.tiles import (
    align_sample_and_tile,
    approximate_num_bytes,
    get_tile_mask,
    num_bytes_without_compression,
    num_tiles_for_sample,
)
from hub.core.fast_forwarding import (
    ffw_chunk_id_encoder,
    ffw_tensor_meta,
    ffw_tile_encoder,
)
import warnings
from hub.util.casting import get_dtype, intelligent_cast
from hub.core.compression import decompress_array, get_compression_factor
from hub.compression import get_compression_type, BYTE_COMPRESSION, IMAGE_COMPRESSION
from math import ceil
from typing import Optional, Sequence, Union, Tuple, List, Set
from hub.util.exceptions import (
    CannotInferTilesError,
    CorruptedMetaError,
    CorruptedSampleError,
    DynamicTensorNumpyError,
    InvalidSubsliceUpdateShapeError,
)
from hub.core.meta.tensor_meta import TensorMeta
from hub.core.index.index import Index, IndexEntry
from hub.core.storage.lru_cache import LRUCache
from hub.core.chunk import Chunk
from hub.core.meta.encode.chunk_id import ChunkIdEncoder
from hub.core.serialize import serialize_input_samples
from hub.core.compression import compress_multiple, decompress_multiple

from hub.util.keys import (
    get_chunk_key,
    get_chunk_id_encoder_key,
    get_tile_encoder_key,
    get_tensor_meta_key,
)
from hub.core.sample import Sample, SampleValue  # type: ignore
from hub.constants import DEFAULT_MAX_CHUNK_SIZE, ENCODING_DTYPE

import numpy as np

from hub.core.storage.lru_cache import LRUCache

from hub.core.chunk import Buffer, Chunk

from hub.core.meta.encode.chunk_id import ChunkIdEncoder
from hub.core.meta.encode.tile import TileEncoder

from hub.core.serialize import serialize_input_sample


SampleValue = Union[np.ndarray, int, float, bool, Sample]


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
            meta_cache (LRUCache): Cache used for storing non chunk data such as tensor meta and chunk id encoder during transforms in memory.

        Raises:
            ValueError: If invalid max chunk size.
        """

        self.key = key
        self.cache = cache
        self._meta_cache = meta_cache
        self.tile_optimizer = TileOptimizer(self.min_chunk_size, self.max_chunk_size, self.tensor_meta)

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

        self._warned_about_suboptimal_chunks = False

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

        key = get_chunk_id_encoder_key(self.key)
        if not self.chunk_id_encoder_exists:
            enc = ChunkIdEncoder()
            self.meta_cache[key] = enc
            return enc

        enc = self.meta_cache.get_cachable(key, ChunkIdEncoder)
        return enc

    @property
    def chunk_id_encoder_exists(self) -> bool:
        try:
            key = get_chunk_id_encoder_key(self.key)
            self.meta_cache[key]
            return True
        except KeyError:
            return False

    @property
    def tile_encoder(self) -> TileEncoder:
        """Gets the tile encoder from cache, if one is not found it creates a blank encoder."""

        key = get_tile_encoder_key(self.key)
        if not key in self.meta_cache:
            enc = TileEncoder()
            self.meta_cache[key] = enc
            return enc

        enc = self.meta_cache.get_cachable(key, TileEncoder)
        return enc

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

        return self.get_chunk(self.last_chunk_key)

    def get_chunk(self, chunk_key: str) -> Chunk:
        return self.cache.get_cachable(chunk_key, Chunk)

    @property
    def last_chunk_key(self) -> str:
        last_chunk_name = self.chunk_id_encoder.get_name_for_chunk(-1)
        last_chunk_key = get_chunk_key(self.key, last_chunk_name)
        return last_chunk_key

    @property
    def tensor_meta(self):
        tensor_meta_key = get_tensor_meta_key(self.key)
        return self.meta_cache.get_cachable(tensor_meta_key, TensorMeta)

    def _needs_multiple_chunks(self, nbytes: int) -> bool:
        """Checks if the last chunk (if it exists) has room for `nbytes`. If not, 
        check if it can fit in a single chunk or multiple."""

        if self.last_chunk is None:
            return nbytes > self.max_chunk_size

        last_has_space = self.last_chunk.has_space_for(nbytes, self.max_chunk_size)
        if not last_has_space:
            if nbytes < self.max_chunk_size:
                return False
            return True

        return False

    def _extend_bytes(
        self,
        buffer: memoryview,
        nbytes: List[int],
        shapes: List[Tuple[int]],
    ):
        """Treat `buffer` as multiple samples and place them into compressed `Chunk`s."""
        if self.tensor_meta.chunk_compression:
            raise NotImplementedError(
                "_extend_bytes not implemented for tensors with chunk wise compression. Use _append_bytes instead."
            )
            
        chunk = self.last_chunk
        new_chunk = self._create_new_chunk

        if chunk is None or self._is_last_chunk_a_tile():
            chunk = new_chunk()

        # If the first incoming sample can't fit in the last chunk, create a new chunk.
        if nbytes[0] > self.min_chunk_size - chunk.num_data_bytes:
            chunk = new_chunk()

        max_chunk_size = self.max_chunk_size
        min_chunk_size = self.min_chunk_size
        enc = self.chunk_id_encoder

        while nbytes:  # len(nbytes) is initially same as number of incoming samples.
            num_samples_to_current_chunk = 0
            nbytes_to_current_chunk = 0
            need_to_tile = False

            for nb in nbytes:  # len(nbytes) = samples remaining to be added to a chunk
                if self._needs_multiple_chunks(nb):
                    if num_samples_to_current_chunk > 0:
                        break

                    need_to_tile = True
                    num_samples_to_current_chunk += 1
                    nbytes_to_current_chunk += nb
                    break

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

            current_buffer = buffer[:nbytes_to_current_chunk]
            current_shapes = shapes[:num_samples_to_current_chunk]
            current_nbytes = nbytes[:num_samples_to_current_chunk]
            
            if need_to_tile:
                # TODO: create tiles with buffers
                raise NotImplementedError
            else:
                chunk.extend_samples(  # type: ignore
                    current_buffer,
                    max_chunk_size,
                    current_shapes,
                    current_nbytes,
                )
                enc.register_samples(num_samples_to_current_chunk)

            # Remove bytes from buffer that have been added to current chunk
            buffer = buffer[nbytes_to_current_chunk:]

            # Remove shapes and nbytes for samples that have beed added to current chunk
            del nbytes[:num_samples_to_current_chunk]
            del shapes[:num_samples_to_current_chunk]

            if buffer:
                chunk = new_chunk()

    def _append_bytes_to_compressed_chunk(self, buffer: memoryview, shape: Tuple[int, ...]):
        """Treat `buffer` as single sample and place them into compressed `Chunk`s."""

        tensor_meta = self.tensor_meta

        chunk_compression = tensor_meta.chunk_compression
        last_chunk_uncompressed = self._last_chunk_uncompressed
        
        dtype = tensor_meta.dtype
        if len(buffer) > 0:
            array = np.frombuffer(buffer, dtype=dtype).reshape(shape)
        else:
            array = np.zeros(shape, dtype=dtype)

        # Append incoming buffer to last chunk and compress:
        last_chunk_uncompressed.append(array)
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

    def _append_bytes(self, buffer: memoryview, shape: Tuple[int, ...], num_samples: int=1):
        """Treat `buffer` as a single sample and place them into `Chunk`s. This function implements the algorithm for
        determining which chunks contain which parts of `buffer`.

        Args:
            buffer (Buffer): Buffer that represents a single sample. Can have a
                length of 0, in which case `shape` should contain at least one 0 (empty sample).
            shape (Tuple[int, ...]): Shape for the sample that `buffer` represents.
        """

        if self.tensor_meta.chunk_compression:
            self._append_bytes_to_compressed_chunk(buffer, shape)
        else:
            buffer_consumed = self._try_appending_to_last_chunk(buffer, shape)
            if not buffer_consumed:
                self._append_to_new_chunk(buffer, shape)

        self.chunk_id_encoder.register_samples(num_samples)

    def _can_set_to_last_chunk(self, nbytes: int) -> bool:
        """Whether last chunk's data can be set to a buffer of size nbytes."""
        last_chunk = self.last_chunk
        if last_chunk is None:
            return False

        if self._is_last_chunk_a_tile():
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

        # synchronize tensor meta
        tensor_meta_key = get_tensor_meta_key(self.key)
        self.meta_cache[tensor_meta_key] = self.tensor_meta

        # synchronize chunk ID encoder
        chunk_id_key = get_chunk_id_encoder_key(self.key)
        self.meta_cache[chunk_id_key] = self.chunk_id_encoder

        # synchronize tile encoder
        tile_encoder_key = get_tile_encoder_key(self.key)
        self.meta_cache[tile_encoder_key] = self.tile_encoder

    def _is_last_chunk_a_tile(self):
        # -2 because we increment tensor meta length before this function is called
        return self.tensor_meta.length - 2 in self.tile_encoder

    def _try_appending_to_last_chunk(
        self, buffer: Buffer, shape: Tuple[int, ...]
    ) -> bool:
        """Will store `buffer` inside of the last chunk if it can.
        It can be stored in the last chunk if it exists and has space for `buffer`.

        Args:
            buffer (Buffer): Data to store. This can represent any number of samples.
            shape (Tuple[int, ...]): Shape for the sample that `buffer` represents.

        Returns:
            bool: True if `buffer` was successfully written to the last chunk, otherwise False.
        """

        last_chunk = self.last_chunk
        if last_chunk is None:
            return False

        # can never append new samples to a tile chunk
        if self._is_last_chunk_a_tile():
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
                return True

        return False

    def _append_to_new_chunk(self, buffer: Buffer, shape: Tuple[int, ...]) -> Chunk:
        """Will create a new chunk and store `buffer` inside of it. Assumes that `buffer`'s length is < max chunk size.
        This should be called if `buffer` could not be added to the last chunk.

        Args:
            buffer (Buffer): Data to store. This can represent any number of samples.
            shape (Tuple[int, ...]): Shape for the sample that `buffer` represents.

        Returns:
            Chunk: The newly created chunk instance.
        """

        # check if `last_chunk_extended` to handle empty samples
        new_chunk = self._create_new_chunk()
        new_chunk.append_sample(buffer, self.max_chunk_size, shape)
        return new_chunk

    def _create_new_chunk(self):
        """Creates and returns a new `Chunk`. Automatically creates an ID for it and puts a reference in the cache."""

        chunk_id = self.chunk_id_encoder.generate_chunk_id()
        chunk = Chunk()
        chunk_name = chunk_name_from_id(chunk_id)
        chunk_key = get_chunk_key(self.key, chunk_name)
        self.cache[chunk_key] = chunk

        # TODO: make these actual properties of the Chunk class
        chunk.name = chunk_name
        chunk.id = chunk_id

        return chunk

    def _update_tensor_meta(self, shape: Tuple[int, ...], num_new_samples: int):
        tensor_meta = self.tensor_meta
        tensor_meta.update_shape_interval(shape)
        tensor_meta.length += num_new_samples

    def extend(self, samples: Union[np.ndarray, Sequence[SampleValue]]):
        """Formats a batch of `samples` and feeds them into `_append_bytes`."""

        self.cache.check_readonly()
        ffw_chunk_id_encoder(self.chunk_id_encoder)

        tensor_meta = self.tensor_meta
        if tensor_meta.dtype is None:
            tensor_meta.set_dtype(get_dtype(samples))

        # TODO: fix logic after merge conflicts
        # CURRENT:
        # for sample in samples:
        #     buffer, shape = serialize_input_sample(sample, tensor_meta)
        #     # TODO: if buffer exceeds a single chunk, use `extend_empty`!

        #     # update tensor meta length first because erroneous meta information is better than un-accounted for data.
        #     self._update_tensor_meta(shape, 1)

        #     if self._needs_multiple_chunks(len(buffer)):
        #         # TODO: optimize tiling for append/extend (sample gets serialized twice!)
        #         self.create_tiles(shape, increment_length=False)

        #         update_index = Index([IndexEntry(-1)])

        #         # use retile=False so we can pass in a sigle-dim effective index 
        #         # TODO: implement re-tiling
        #         self.update(update_index, sample, retile=False)

        #     else:
        #         self._append_bytes(buffer, shape)

        # INCOMING
        buff, nbytes, shapes = serialize_input_samples(
            samples, tensor_meta
        )

        for shape in shapes:
            tensor_meta.update_shape_interval(shape)
        tensor_meta.length += len(samples)
        
        if tensor_meta.chunk_compression:
            for nb, shape in zip(nbytes, shapes):
                current_buffer = buff[:nb]
                current_shape = shape[:]

                if self._needs_multiple_chunks(len(current_buffer)):
                    raise NotImplementedError  # TODO
                    tiled_buffers = None
                    self.create_tiles(current_shape, increment_length=False, buffers=tiled_buffers)
                else:
                    self._append_bytes(current_buffer, current_shape)  # type: ignore

                buff = buff[nb:]
        else:
            self._extend_bytes(buff, nbytes, shapes[:])  # type: ignore

        self._synchronize_cache()
        self.cache.maybe_flush()

    def append(self, sample: SampleValue):
        """Formats a single `sample` (compresseses/decompresses if applicable) and feeds it into `_append_bytes`."""

        self.extend([sample])

    def extend_empty(self, shape: Tuple[int, ...]):
        """Create an empty sample with `shape`. If `shape` exceeds a single chunk, a set of tiled placeholder chunks will be created.
        These placeholder tile chunks can be filled with actual data later by updating."""

        sample_shape = shape[1:]
        for _ in range(shape[0]):
            self.create_tiles(sample_shape)

    def update(
        self,
        index: Index,
        incoming_samples: Union[Sequence[SampleValue], SampleValue],
        operator: Optional[str] = None,
        retile: bool=True,
    ):
        """Update data at `index` with `samples`."""
        # TODO: docstring

        # TODO: break into smaller functions

        if operator is not None:
            return self._update_with_operator(index, incoming_samples, operator)

        self.cache.check_readonly()
        tensor_meta = self.tensor_meta
        tile_encoder = self.tile_encoder
        chunk_id_encoder = self.chunk_id_encoder

        chunk_compression = tensor_meta.chunk_compression
        dtype = tensor_meta.dtype

        ffw_chunk_id_encoder(self.chunk_id_encoder)
        ffw_tensor_meta(tensor_meta)
        ffw_tile_encoder(tile_encoder)

        value0_index, subslice_index = index.split_subslice()
        is_full_sample_replacement = index.is_single_dim_effective()

        # TODO: refac this
        index_length: int = value0_index.shape[0]  # type: ignore
        if index_length is None:
            index_length = tensor_meta.length

        incoming_samples = _make_sequence(incoming_samples, index_length)

        chunks_nbytes_after_updates = []

        # update one sample at a time
        iterator = value0_index.values[0].indices(self.num_samples)
        for i, global_sample_index in enumerate(iterator):
            incoming_sample = incoming_samples[i]

            if isinstance(incoming_sample, Sample):
                incoming_sample = incoming_sample.array

            if not isinstance(incoming_sample, np.ndarray):
                incoming_sample = np.asarray(incoming_sample).astype(dtype)

            local_sample_index = chunk_id_encoder.translate_index_relative_to_chunks(
                global_sample_index
            )

            tiles = self.download_required_tiles(
                global_sample_index, subslice_index
            )

            is_tiled = tiles.size > 1

            if retile and is_full_sample_replacement and is_tiled:
                # TODO: implement sample re-tiling
                raise NotImplementedError(
                    "Re-tiling samples is not yet supported!"
                )

            for tile_index, tile_object in np.ndenumerate(tiles):
                if tile_object is None:
                    continue

                if is_full_sample_replacement:
                    # no need to read the sample, just purely replace
                    new_sample = incoming_sample

                else:

                    tile = self.read_sample_from_chunk(global_sample_index, tile_object)
                    tile = np.array(
                        tile
                    )  # memcopy necessary to support inplace updates using numpy slicing

                    if is_tiled:
                        tile_shape_mask = tile_encoder.get_tile_shape_mask(global_sample_index, tiles)
                        full_sample_shape = tile_encoder.get_sample_shape(global_sample_index)

                        # sanity check
                        tile_shape = tile_shape_mask[tile_index]
                        if tile.shape != tile_shape:
                            raise CorruptedSampleError(
                                f"Tile encoder has the incorrect tile shape. Tile shape: {tile.shape}, tile encoder shape: {tile_shape}"
                            )
                    else:
                        tile_index = None
                        full_sample_shape = tile.shape
                    
                    expected_subslice_shape = subslice_index.shape_if_applied_to(full_sample_shape, squeeze=True)
                    if expected_subslice_shape != incoming_sample.shape:
                        raise InvalidSubsliceUpdateShapeError(incoming_sample.shape, expected_subslice_shape)

                    tile_view, incoming_sample_view = align_sample_and_tile(incoming_sample, tile, subslice_index, tile_index)
                    tile_view[:] = incoming_sample_view
                    new_sample = tile

                buffer, shape = serialize_input_sample(new_sample, tensor_meta)
                tile_object.update_sample(local_sample_index, buffer, shape, chunk_compression=chunk_compression, dtype=dtype)

                if is_full_sample_replacement:
                    self._update_tensor_meta(shape, 0)

                # we only care about warning regarding updates for non-finalized chunks
                check_bytes = global_sample_index != tensor_meta.length - 1 and not is_tiled
                if check_bytes:
                    chunks_nbytes_after_updates.append(tile_object.nbytes)

            # don't spam warnings
            if not self._warned_about_suboptimal_chunks:
                warned = _warn_if_suboptimal_chunks(chunks_nbytes_after_updates, self.min_chunk_size, self.max_chunk_size, is_tiled)
                if warned:
                    self._warned_about_suboptimal_chunks = warned

            self._synchronize_cache()  # TODO: refac, sync metas + sync tiles separately
            self._sync_tiles(tiles)
            self.cache.maybe_flush()
            self.meta_cache.maybe_flush()


    def create_tiles(self, sample_shape: Tuple[int, ...], increment_length: bool=True, buffer: Buffer=None):
        # TODO: docstring (mention buffer should be compressed)

        self.cache.check_readonly()
        ffw_chunk_id_encoder(self.chunk_id_encoder)

        tensor_meta = self.tensor_meta
        dtype = tensor_meta.dtype
        if dtype is None:
            raise CannotInferTilesError(
                "Cannot add an empty sample to a tensor with dtype=None. Either add a real sample, or use `tensor.set_dtype(...)` first."
            )

        # TODO: functionize this
        if buffer is None:
            # for `append_empty`, we can only approximate
            compression_factor = get_compression_factor(tensor_meta)
            nbytes = approximate_num_bytes(sample_shape, tensor_meta.dtype, compression_factor)
        else:
            nbytes = len(buffer)
            compression_factor = num_bytes_without_compression(sample_shape, dtype) // nbytes

        if self._needs_multiple_chunks(nbytes):
            tile_shape = self.tile_optimizer.optimize(sample_shape, compression_factor)
            num_tiles = num_tiles_for_sample(tile_shape, sample_shape)

            idx = self.num_samples
            tile_encoder = self.tile_encoder

            tile_encoder.register_sample(idx, sample_shape, tile_shape)

            if increment_length:
                self._update_tensor_meta(sample_shape, 1)

            # tile_layout = np.empty(tile_encoder.get_tile_layout_shape(idx))
            # for tile_index, _ in np.ndenumerate(tile_layout):
            #     print(tile_index)

            # initialize our N empty chunks including headers
            for i in range(num_tiles):
                self._create_new_chunk()
                empty_buffer = memoryview(bytes())

                # TODO: explain
                self._append_bytes(empty_buffer, tile_shape, num_samples=0 if i > 0 else 1)

            # TODO: can probably get rid of tile encoder meta if we can store `tile_shape` inside of the chunk's ID!

            # TODO: make sure that the next appended/extended sample does NOT get added to the last tile chunk that is created by this method!!!!
            self._synchronize_cache()
            self.cache.maybe_flush()
        else:
            if buffer is not None:
                raise NotImplementedError("Calling `create_tiles` can't be done with a buffer when the sample fits in a single chunk.")

            empty_sample = np.zeros(sample_shape, dtype=tensor_meta.dtype)
            self.append(empty_sample)

    def sample_from_tiles(
        self, global_sample_index: int, subslice_index: Index, dtype: np.dtype
    ) -> np.ndarray:
        # TODO: docstring

        tiles = self.download_required_tiles(
            global_sample_index, subslice_index
        )
        sample = self.coalesce_sample(
            global_sample_index, tiles, subslice_index, dtype
        )

        return sample

    def _sync_tiles(self, tiles: np.ndarray):
        # TODO: docstring

        for _, tile in np.ndenumerate(tiles):
            if tile is not None:
                self.cache[tile.key] = tile

    def download_required_tiles(
        self, global_sample_index: int, subslice_index: Index
    ) -> np.ndarray:
        # TODO: docstring

        chunk_id_encoder = self.chunk_id_encoder
        tile_encoder = self.tile_encoder

        tile_ids = chunk_id_encoder[global_sample_index]

        ordered_tile_ids = tile_encoder.order_tiles(global_sample_index, tile_ids)  # type: ignore
        tile_shape_mask = tile_encoder.get_tile_shape_mask(
            global_sample_index, ordered_tile_ids
        )
        tile_mask = get_tile_mask(ordered_tile_ids, tile_shape_mask, subslice_index)
        tiles = self.download_tiles(ordered_tile_ids, tile_mask)

        return tiles

    def download_tiles(
        self, ordered_tile_ids: np.ndarray, download_mask: np.ndarray
    ) -> np.ndarray:
        """Downloads the tiles and returns a numpy array of Chunk objects with the same shape.

        Args:
            ordered_tile_ids (np.ndarray): Array of tile (chunk) IDs with their shape in tile-order.
            download_mask (np.ndarray): Boolean array with the same shape of `ordered_tile_ids`.
                If the corresponding element is `True`, the chunk with it's ID will be downloaded.

        Raises:
            ValueError: If the shape of `ordered_tile_ids` and `download_mask` do not match.

        Returns:
            # TODO
        """

        if ordered_tile_ids.shape != download_mask.shape:
            raise ValueError(
                f"Tiles {ordered_tile_ids.shape} and the download mask {download_mask.shape} should be the same shape."
            )

        chunks = np.empty(ordered_tile_ids.shape, dtype=object)

        for tile_index, tile_id in np.ndenumerate(ordered_tile_ids):
            need_tile = download_mask[tile_index]

            if need_tile:
                tile = self.download_tile(tile_id)
                chunks[tile_index] = tile

        return chunks

    def download_tile(self, tile_id: ENCODING_DTYPE) -> Chunk:
        # TODO: docstring

        chunk_name = chunk_name_from_id(tile_id)
        chunk_key = get_chunk_key(self.key, chunk_name)
        chunk = self.cache.get_cachable(chunk_key, Chunk)
        chunk.key = chunk_key
        return chunk

    def coalesce_sample(
        self,
        global_sample_index: int,
        tiles: np.ndarray,
        subslice_index: Index,
        dtype: np.dtype,
    ) -> np.ndarray:
        # TODO: docstring

        # TODO: break into smaller methods

        is_tiled = tiles.size > 1

        if not is_tiled:
            sample = self.read_sample_from_chunk(global_sample_index, tiles[0])
            return subslice_index.apply([sample], include_first_value=True)[0]

        tile_encoder = self.tile_encoder
        full_sample_shape = tile_encoder.get_sample_shape(global_sample_index)
        sample_shape = subslice_index.shape_if_applied_to(full_sample_shape)
        sample = np.zeros(sample_shape, dtype=dtype)

        for tile_index, tile_obj in np.ndenumerate(tiles):
            if tile_obj is None:
                continue
            tile = self.read_sample_from_chunk(global_sample_index, tile_obj)
            tile_view, sample_view = align_sample_and_tile(sample, tile, subslice_index, tile_index)
            sample_view[:] = tile_view

        return sample

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
    ) -> Union[np.ndarray, Sequence[np.ndarray]]:
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
        last_shape = None
        samples = []

        tensor_meta = self.tensor_meta
        dtype = tensor_meta.dtype

        value0_index, subslice_index = index.split_subslice()

        for global_sample_index in value0_index.values[0].indices(length):
            sample = self.sample_from_tiles(global_sample_index, subslice_index, dtype)
            shape = sample.shape

            if not aslist and last_shape is not None:
                if shape != last_shape:
                    raise DynamicTensorNumpyError(self.key, index, "shape")

            samples.append(sample)
            last_shape = shape

        return _format_read_samples(samples, index, aslist)

    def get_chunk_from_id(self, chunk_id: ENCODING_DTYPE):
        chunk_name = chunk_name_from_id(chunk_id)
        chunk_key = get_chunk_key(self.key, chunk_name)
        chunk = self.cache.get_cachable(chunk_key, Chunk)
        chunk.key = chunk_key
        return chunk

    def read_sample_from_chunk(
        self, global_sample_index: int, chunk: Chunk, cast: bool = True, copy=False
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
        if chunk_compression:
            if get_compression_type(chunk_compression) == BYTE_COMPRESSION:
                decompressed = chunk.decompressed_data(compression=chunk_compression)
                sb, eb = chunk.byte_positions_encoder[local_sample_index]
                buffer = decompressed[sb:eb]

                if len(buffer) > 0:
                    array = np.frombuffer(buffer, dtype=dtype).reshape(shape)
                else:
                    array = np.zeros(shape, dtype=dtype)
                
                return array
            else:
                return chunk.decompressed_samples()[local_sample_index]
        sb, eb = chunk.byte_positions_encoder[local_sample_index]
        buffer = buffer[sb:eb]

        sample_compression = self.tensor_meta.sample_compression
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
            chunk = chunk_name_from_id(chunk_id)
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
            tkey = get_tensor_meta_key(self.key)
            ikey = get_chunk_id_encoder_key(self.key)
            raise CorruptedMetaError(
                f"'{tkey}' and '{ikey}' have a record of different numbers of samples. Got {tensor_meta_length} and {chunk_id_num_samples} respectively."
            )


def _format_read_samples(
    samples: Sequence[np.array], index: Index, aslist: bool
) -> Union[np.ndarray, List[np.ndarray]]:
    """Prepares samples being read from the chunk engine in the format the user expects."""

    if aslist and all(map(np.isscalar, samples)):
        samples = list(arr.item() for arr in samples)

    samples = index.apply_squeeze(samples)  # type: ignore

    if aslist:
        return samples
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
    chunks_nbytes_after_updates: List[int], min_chunk_size: int, max_chunk_size: int, is_tiled: bool
) -> bool:
    """Returns True if warning executed."""

    upper_warn_threshold = max_chunk_size * (1 + CHUNK_UPDATE_WARN_PORTION)
    lower_warn_threshold = min_chunk_size * (1 - CHUNK_UPDATE_WARN_PORTION)

    for nbytes in chunks_nbytes_after_updates:
        too_large = nbytes > upper_warn_threshold

        # TODO: tiled samples need custom policy for warning when the chunk size is lower than
        # min chunk size. this is because they are initialized as all zeros.
        too_small = False
        if not is_tiled:
            too_small = nbytes < lower_warn_threshold

        if too_large or too_small:
            reason = ""
            
            if too_large:
                reason = f"too large, {nbytes} > {upper_warn_threshold}"
            else:
                reason = f"too small, {nbytes} < {lower_warn_threshold}"

            warnings.warn(
                f"After update, some chunks were suboptimal ({reason}). Be careful when doing lots of updates that modify the sizes of samples by a large amount, these can heavily impact read performance!"
            )
            return True
    return False
