import deeplake
from deeplake.core.chunk.base_chunk import BaseChunk
from deeplake.core.chunk_engine import ChunkEngine
from deeplake.core.chunk.uncompressed_chunk import UncompressedChunk
from deeplake.core.compression import _read_video_shape, _decompress_video
from deeplake.core.index.index import Index
from deeplake.core.link_creds import LinkCreds
from deeplake.core.linked_sample import LinkedSample
from deeplake.core.meta.encode.chunk_id import ChunkIdEncoder
from deeplake.core.meta.encode.creds import CredsEncoder
from deeplake.core.storage import LRUCache
from deeplake.core.tensor_link import read_linked_sample
from deeplake.core.tiling.deserialize import (
    coalesce_tiles,
    np_list_to_sample,
    translate_slices,
)
from deeplake.core.linked_sample import read_linked_sample
from deeplake.util.exceptions import (
    BadLinkError,
    GetDataFromLinkError,
    ReadOnlyModeError,
)
from deeplake.util.keys import get_creds_encoder_key
from deeplake.util.link import get_path_creds_key, save_link_creds
from deeplake.util.video import normalize_index
from deeplake.util.path import get_path_type
import numpy as np
from typing import Optional, Dict, Any, Tuple, Union
from PIL import Image  # type: ignore
from deeplake.core.linked_tiled_sample import LinkedTiledSample
from math import ceil


def remove_chunk_engine_compression(chunk_engine):
    chunk_engine.chunk_class = UncompressedChunk
    chunk_engine.compression = None
    chunk_engine._sample_compression = None
    chunk_engine._chunk_compression = None


class LinkedChunkEngine(ChunkEngine):
    def __init__(
        self,
        key: str,
        cache: LRUCache,
        version_state: Dict[str, Any],
        link_creds: LinkCreds,
        meta_cache: Optional[LRUCache] = None,
    ):
        super().__init__(key, cache, version_state, meta_cache)
        self.path_chunk_engine = ChunkEngine(key, cache, version_state, meta_cache)
        remove_chunk_engine_compression(self)
        remove_chunk_engine_compression(self.path_chunk_engine)
        self.link_creds = link_creds  # type: ignore
        self._creds_encoder: Optional[CredsEncoder] = None
        self._creds_encoder_commit_id: Optional[str] = None

    @property
    def creds_encoder(self) -> CredsEncoder:
        commit_id = self.commit_id
        if self._creds_encoder is None or self._creds_encoder_commit_id != commit_id:
            commit_id = self.commit_id
            key = get_creds_encoder_key(self.key, commit_id)
            if not self.creds_encoder_exists:
                enc = CredsEncoder()
                try:
                    self.meta_cache[key] = enc
                except ReadOnlyModeError:
                    pass
            else:
                enc = self.meta_cache.get_deeplake_object(key, CredsEncoder)
            self._creds_encoder = enc
            self._creds_encoder_commit_id = commit_id
            self.meta_cache.register_deeplake_object(key, enc)
        return self._creds_encoder

    @property
    def creds_encoder_exists(self):
        commit_id = self.commit_id
        if (
            self._creds_encoder is not None
            and self._creds_encoder_commit_id == commit_id
        ):
            return True
        try:
            key = get_creds_encoder_key(self.key, commit_id)
            self.meta_cache[key]
            return True
        except KeyError:
            return False

    @property
    def is_data_cachable(self):
        return False

    def linked_sample(
        self, global_sample_index: int
    ) -> Union[LinkedSample, LinkedTiledSample]:
        sample_creds_key = self.creds_key(global_sample_index)
        if self._is_tiled_sample(global_sample_index):
            path_array: np.ndarray = (
                super()
                .get_basic_sample(
                    global_sample_index,
                    Index(global_sample_index),
                    fetch_chunks=True,
                    is_tile=True,
                )
                .path_array
            )
            return LinkedTiledSample(path_array, sample_creds_key)
        sample_path = self.get_path(global_sample_index, fetch_chunks=True)
        return LinkedSample(sample_path, sample_creds_key)

    def creds_key(self, global_sample_index: int):
        sample_creds_encoded = self.creds_encoder.get_encoded_creds_key(
            global_sample_index
        )
        return self.link_creds.get_creds_key(sample_creds_encoded)  # type: ignore

    def get_video_url(self, global_sample_index):
        sample_path = self.get_path(global_sample_index)
        sample_creds_key = self.creds_key(global_sample_index)
        storage = None
        if sample_path.startswith(
            ("gcs://", "gcp://", "gs://", "s3://", "az://", "azure://")
        ):
            provider_type = get_path_type(sample_path)
            storage = self.link_creds.get_storage_provider(
                sample_creds_key, provider_type
            )
            url = storage.get_presigned_url(sample_path, full=True)
        else:
            url = sample_path
        return url, sample_path

    def get_video_sample(self, global_sample_index, index, decompress=True):
        url, path = self.get_video_url(global_sample_index)
        try:
            squeeze = isinstance(index, int)
            shape = _read_video_shape(url)
            sub_index = index.values[1].value if len(index.values) > 1 else None  # type: ignore
            start, stop, step, reverse = normalize_index(sub_index, shape[0])
            video_sample = _decompress_video(
                url,
                start,
                stop,
                step,
                reverse,
            )
            if squeeze:
                video_sample.squeeze(0)
            return video_sample
        except Exception as e:
            raise GetDataFromLinkError(path)

    def get_full_tiled_sample(
        self, global_sample_index: int, fetch_chunks: bool = False
    ):
        tile_enc = self.tile_encoder
        shape = tile_enc.get_sample_shape(global_sample_index)
        tile_shape = tile_enc.get_tile_shape(global_sample_index)
        layout_shape = tile_enc.get_tile_layout_shape(global_sample_index)
        path_array: np.ndarray = (
            super()
            .get_basic_sample(
                global_sample_index,
                Index(global_sample_index),
                fetch_chunks,
                is_tile=True,
            )
            .path_array
        )

        sample_creds_key = self.creds_key(global_sample_index)
        tiled_arrays = [
            read_linked_sample(path, sample_creds_key, self.link_creds, False).array
            for path in iter(path_array.flat)
        ]
        return np_list_to_sample(tiled_arrays, shape, tile_shape, layout_shape)

    def get_partial_tiled_sample(self, global_sample_index, index, fetch_chunks=False):
        tile_enc = self.tile_encoder
        sample_shape = tile_enc.get_sample_shape(global_sample_index)
        tile_shape = tile_enc.get_tile_shape(global_sample_index)
        ordered_tile_paths = (
            super()
            .get_basic_sample(
                global_sample_index,
                Index(global_sample_index),
                fetch_chunks,
                is_tile=True,
            )
            .path_array
        )
        tiles_index, sample_index = translate_slices(
            [v.value for v in index.values[1:]], sample_shape, tile_shape  # type: ignore
        )
        required_tile_paths = ordered_tile_paths[tiles_index]

        sample_creds_key = self.creds_key(global_sample_index)

        tiles = np.vectorize(
            lambda path: read_linked_sample(
                path, sample_creds_key, self.link_creds, False
            ).array,
            otypes=[object],
        )(required_tile_paths)
        sample = coalesce_tiles(tiles, tile_shape, None)
        sample = sample[sample_index]
        return sample

    def get_basic_sample(
        self,
        global_sample_index,
        index,
        fetch_chunks=False,
        is_tile=False,
        decompress=True,
    ):
        sample = self.get_deeplake_read_sample(global_sample_index, fetch_chunks)
        if sample is None:
            return np.ones((0,))
        arr = sample.array
        max_shape = self.tensor_meta.max_shape
        if len(arr.shape) == 2 and max_shape and len(max_shape) == 3:
            arr = arr.reshape(arr.shape + (1,))
        return arr[tuple(entry.value for entry in index.values[1:])]

    def get_path(self, global_sample_index, fetch_chunks=False) -> str:
        return super().get_basic_sample(
            global_sample_index, Index(global_sample_index), fetch_chunks
        )[0]

    def get_deeplake_read_sample(self, global_sample_index, fetch_chunks=False):
        creds_encoder = self.creds_encoder
        sample_path = self.get_path(global_sample_index, fetch_chunks)
        if not sample_path:
            return None
        sample_creds_key = self.creds_key(global_sample_index)
        return read_linked_sample(sample_path, sample_creds_key, self.link_creds, False)

    @property
    def verify(self):
        return self.tensor_meta.is_link and self.tensor_meta.verify

    def check_each_sample(self, samples, verify=True, ignore_errors=False):
        link_creds = self.link_creds
        verified_samples = []
        skipped = []
        for i, sample in enumerate(samples):
            try:
                if isinstance(sample, deeplake.core.tensor.Tensor) and sample.is_link:
                    sample = sample._linked_sample()
                    samples[i] = sample
                elif (
                    not isinstance(sample, (LinkedSample, LinkedTiledSample))
                    and sample is not None
                ):
                    raise TypeError(
                        f"Expected LinkedSample or LinkedTiledSample, got {type(sample)} instead. Use deeplake.link() to link samples or deeplake.link_tiled() to link multiple images as tiles."
                    )

                path, creds_key = get_path_creds_key(sample)

                # verifies existence of creds_key
                if verify:
                    link_creds.get_encoding(creds_key, path)

                if sample is None or sample.path == "":
                    verified_samples.append(sample)
                elif isinstance(sample, LinkedTiledSample):
                    verify_samples = self.verify and verify
                    sample.set_check_tile_shape(self.link_creds, verify_samples)
                    sample.set_sample_shape()
                    verified_samples.append(sample)
                else:
                    try:
                        _verify = verify and self.verify
                        verified_samples.append(
                            read_linked_sample(
                                sample.path,
                                sample.creds_key,
                                self.link_creds,
                                verify=_verify,
                            )
                        )
                    except Exception as e:
                        raise BadLinkError(sample.path, sample.creds_key) from e
            except Exception:
                if ignore_errors:
                    skipped.append(i)
                    continue
                raise

        for i in reversed(skipped):
            samples.pop(i)

        return verified_samples

    def register_new_creds(self, num_samples_added, samples):
        num_samples_added = ceil(num_samples_added)
        link_creds = self.link_creds
        creds_encoder = self.creds_encoder
        for i in range(num_samples_added):
            sample = samples[i]
            path, creds_key = get_path_creds_key(sample)
            encoded_creds_key = link_creds.get_encoding(creds_key, path)
            creds_encoder.register_samples((encoded_creds_key,), 1)
            if link_creds.add_to_used_creds(creds_key):
                save_link_creds(self.link_creds, self.cache)
                self.link_creds.warn_missing_managed_creds()

    def update_creds(
        self,
        sample_index: int,
        sample: Optional[Union[LinkedSample, LinkedTiledSample]],
    ):
        link_creds = self.link_creds
        path, creds_key = get_path_creds_key(sample)
        encoded_creds_key = link_creds.get_encoding(creds_key, path)  # type: ignore
        self.creds_encoder[sample_index] = (encoded_creds_key,)
        if link_creds.add_to_used_creds(creds_key):  # type: ignore
            save_link_creds(self.link_creds, self.cache)  # type: ignore
            self.link_creds.warn_missing_managed_creds()  # type: ignore

    def read_shape_for_sample(self, global_sample_index: int) -> Tuple[int, ...]:
        if self._is_tiled_sample(global_sample_index):
            return self.tile_encoder.get_sample_shape(global_sample_index)
        sample = self.get_deeplake_read_sample(global_sample_index)
        if sample is None:
            return (0,)
        return sample.shape

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
        local_sample_index = enc.translate_index_relative_to_chunks(global_sample_index)
        sample_path = chunk.read_sample(
            local_sample_index, cast=cast, copy=copy, decompress=decompress
        )[0]

        if not sample_path:
            return self.get_empty_sample()
        sample_creds_key = self.creds_key(global_sample_index)
        read_sample = read_linked_sample(
            sample_path, sample_creds_key, self.link_creds, False
        )
        if to_pil:
            return read_sample.pil
        return read_sample.array

    def check_link_ready(self):
        missing_keys = self.link_creds.missing_keys
        creds_used = self.link_creds.used_creds_keys
        missing_used_keys = {key for key in missing_keys if key in creds_used}
        if missing_used_keys:
            raise ValueError(
                f"Creds keys {missing_used_keys} are used in the data but not populated. Please populate the dataset using ds.populate_creds()."
            )

    def _pop_from_chunk(self, chunk: Optional[BaseChunk], row: int, global_idx: int):
        self.creds_encoder.pop(global_idx)
        return super()._pop_from_chunk(chunk, row, global_idx)

    def get_empty_sample(self):
        return np.ones((0,))

    def read_bytes_for_sample(self, global_sample_index: int) -> bytes:
        if self._is_tiled_sample(global_sample_index):
            raise ValueError(
                "Cannot read bytes for a link_tiled sample. Please read the sample as a numpy array."
            )
        sample = self.get_deeplake_read_sample(global_sample_index)
        return sample.buffer

    def path(self, index, aslist, fetch_chunks):
        return self.path_chunk_engine.numpy(
            index, aslist=aslist, fetch_chunks=fetch_chunks, use_data_cache=False
        )

    def _update_non_tiled_sample(
        self, global_sample_index: int, index: Index, sample, nbytes_after_updates
    ):
        if len(index.values) != 1:
            raise ValueError(
                "Cannot update a partial value of a linked sample. Please update the entire sample."
            )
        super()._update_non_tiled_sample(
            global_sample_index, index, sample, nbytes_after_updates
        )

    def _update_tiled_sample(
        self, global_sample_index: int, index: Index, sample, nbytes_after_updates
    ):
        self._update_non_tiled_sample(
            global_sample_index, index, sample, nbytes_after_updates
        )

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
        sample: LinkedTiledSample = samples[0]
        if register:
            if start_chunk_row is not None:
                enc.register_samples(1)
            else:
                enc_count[-1] += 1
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
        return num_samples_added, samples, lengths
