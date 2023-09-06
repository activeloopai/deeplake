import numpy as np
from typing import List, Union, Optional
from deeplake.core.linked_tiled_sample import LinkedTiledSample
from deeplake.core.serialize import (
    check_sample_shape,
    bytes_to_text,
    deserialize_linked_tiled_sample,
)
from deeplake.core.tiling.sample_tiles import SampleTiles
from deeplake.core.polygon import Polygons
from deeplake.util.exceptions import TensorDtypeMismatchError
from deeplake.constants import ENCODING_DTYPE
from .base_chunk import BaseChunk, InputSample, catch_chunk_read_error


class UncompressedChunk(BaseChunk):
    def extend_if_has_space(  # type: ignore
        self,
        incoming_samples: Union[List[InputSample], np.ndarray],
        update_tensor_meta: bool = True,
        lengths: Optional[List[int]] = None,
        ignore_errors: bool = False,
    ) -> float:
        self.prepare_for_write()
        if lengths is not None:  # this is triggered only for htype == "text"
            return self._extend_if_has_space_text(
                incoming_samples, update_tensor_meta, lengths
            )
        if isinstance(incoming_samples, np.ndarray):
            if incoming_samples.dtype == object:
                incoming_samples = list(incoming_samples)
            else:
                return self._extend_if_has_space_numpy(
                    incoming_samples, update_tensor_meta
                )
        return self._extend_if_has_space_list(
            incoming_samples, update_tensor_meta, ignore_errors=ignore_errors
        )

    def _extend_if_has_space_text(
        self,
        incoming_samples,
        update_tensor_meta: bool = True,
        lengths=None,
    ) -> float:
        csum = np.cumsum(lengths)
        min_chunk_size = self.min_chunk_size
        num_data_bytes = self.num_data_bytes
        space_left = min_chunk_size - num_data_bytes
        idx = np.searchsorted(csum, space_left)
        if not idx and csum[0] > space_left:
            if self._data_bytes:
                return 0
        num_samples = int(min(len(incoming_samples), idx + 1))  # type: ignore
        bts = list(
            map(self._text_sample_to_byte_string, incoming_samples[:num_samples])
        )
        self._data_bytes += b"".join(bts)  # type: ignore
        bps = np.zeros((num_samples, 3), dtype=ENCODING_DTYPE)
        enc = self.byte_positions_encoder
        arr = enc._encoded
        if len(arr):
            last_seen = arr[-1, 2] + 1
            if len(arr) == 1:
                offset = (arr[0, 2] + 1) * arr[0, 0]
            else:
                offset = (arr[-1, 2] - arr[-2, 2]) * arr[-1, 0] + arr[-1, 1]
        else:
            last_seen = 0
            offset = 0
        bps[:, 2] = np.arange(last_seen, num_samples + last_seen)
        bps[0, 1] = offset
        for i, b in enumerate(bts):
            lengths[i] = len(b)
        lview = lengths[:num_samples]
        csum = np.cumsum(lengths[: num_samples - 1])
        bps[:, 0] = lview
        csum += offset
        bps[1:, 1] = csum
        if len(arr):
            arr = np.concatenate([arr, bps], 0)
        else:
            arr = bps
        enc._encoded = arr
        shape = (1,)
        self.register_sample_to_headers(None, shape, num_samples=num_samples)
        if update_tensor_meta:
            self.update_tensor_meta(shape, num_samples)
        return num_samples

    def _extend_if_has_space_numpy(
        self,
        incoming_samples: np.ndarray,
        update_tensor_meta: bool = True,
    ) -> float:
        num_samples: int
        elem = incoming_samples[0]
        shape = elem.shape
        if not shape:
            shape = (1,)
        chunk_num_dims = self.num_dims
        if chunk_num_dims is None:
            self.num_dims = elem.ndim
        else:
            check_sample_shape(shape, chunk_num_dims)
        size = elem.size
        self.num_dims = self.num_dims or len(shape)
        if size == 0:
            num_samples = len(incoming_samples)
        else:
            num_data_bytes = self.num_data_bytes
            num_samples = max(
                0,
                min(
                    len(incoming_samples),
                    (self.min_chunk_size - num_data_bytes) // elem.nbytes,
                ),
            )
            if not num_samples:
                if num_data_bytes:
                    return 0.0
                else:
                    tiling_threshold = self.tiling_threshold
                    if tiling_threshold < 0 or elem.nbytes < tiling_threshold:
                        num_samples = 1
                    else:
                        return (
                            -1
                        )  # Bail. Chunk engine will try again with incoming_samples as list.
        samples = incoming_samples[:num_samples]
        chunk_dtype = self.dtype
        samples_dtype = incoming_samples.dtype
        if samples_dtype != chunk_dtype:
            if size:
                if not np.can_cast(samples_dtype, chunk_dtype):
                    raise TensorDtypeMismatchError(
                        chunk_dtype,
                        samples_dtype,
                        self.htype,
                    )
            samples = samples.astype(chunk_dtype)
        self._data_bytes += samples.tobytes()  # type: ignore
        self.register_in_meta_and_headers(
            samples[0].nbytes,
            shape,
            update_tensor_meta=update_tensor_meta,
            num_samples=num_samples,
        )
        return num_samples

    def _extend_if_has_space_list(
        self,
        incoming_samples: List[InputSample],
        update_tensor_meta: bool = True,
        ignore_errors: bool = False,
    ) -> float:
        num_samples: float = 0
        skipped: List[int] = []

        for i, incoming_sample in enumerate(incoming_samples):
            try:
                serialized_sample, shape = self.serialize_sample(incoming_sample)
                if shape is not None and not self.tensor_meta.is_link:
                    self.num_dims = self.num_dims or len(shape)
                    check_sample_shape(shape, self.num_dims)
            except Exception as e:
                if ignore_errors:
                    skipped.append(i)
                    continue
                raise

            # NOTE re-chunking logic should not reach to this point, for Tiled ones we do not have this logic
            if isinstance(serialized_sample, SampleTiles):
                incoming_samples[i] = serialized_sample  # type: ignore
                if self.is_empty:
                    self.write_tile(serialized_sample)
                    num_samples += 0.5
                break
            else:
                sample_nbytes = len(serialized_sample)
                if self.is_empty or self.can_fit_sample(sample_nbytes):
                    self._data_bytes += serialized_sample  # type: ignore

                    self.register_in_meta_and_headers(
                        sample_nbytes,
                        shape,
                        update_tensor_meta=update_tensor_meta,
                    )
                    if isinstance(incoming_sample, LinkedTiledSample):
                        num_samples += 0.5
                        break

                    num_samples += 1
                else:
                    break

        for i in reversed(skipped):
            incoming_samples.pop(i)
        return num_samples

    @catch_chunk_read_error
    def read_sample(
        self,
        local_index: int,
        cast: bool = True,
        copy: bool = False,
        sub_index: Optional[Union[int, slice]] = None,
        stream: bool = False,
        decompress: bool = True,
        is_tile: bool = False,
    ):
        self.check_empty_before_read()
        partial_sample_tile = self._get_partial_sample_tile()
        if partial_sample_tile is not None:
            return partial_sample_tile
        buffer = self.memoryview_data
        is_polygon = self.htype == "polygon"
        bps = self.byte_positions_encoder
        if not is_tile and self.is_fixed_shape:
            shape = tuple(self.tensor_meta.min_shape)
            if is_polygon:
                if not bps.is_empty():
                    sb, eb = bps[local_index]
            else:
                sb, eb = self.get_byte_positions(local_index)
            buffer = buffer[sb:eb]
        else:
            if self.tensor_meta.is_link and is_tile:
                return deserialize_linked_tiled_sample(buffer)
            bps_empty = bps.is_empty()
            try:
                shape = self.shapes_encoder[local_index]
            except IndexError as e:
                if not bps_empty:
                    self.num_dims = self.num_dims or len(self.tensor_meta.max_shape)
                    shape = (0,) * self.num_dims
                else:
                    raise e

            if not bps_empty:
                sb, eb = bps[local_index]
                buffer = buffer[sb:eb]

        if self.tensor_meta.htype == "polygon":
            return Polygons.frombuffer(
                bytes(buffer),
                dtype=self.tensor_meta.dtype,
                ndim=shape[-1],
            )

        if not decompress:
            if copy:
                buffer = bytes(buffer)
            return buffer
        if self.is_text_like:
            buffer = bytes(buffer)
            return bytes_to_text(buffer, self.htype)
        ret = np.frombuffer(buffer, dtype=self.dtype).reshape(shape)
        if copy and not ret.flags["WRITEABLE"]:
            ret = ret.copy()
        return ret

    def update_sample(self, local_index: int, sample: InputSample):
        self.prepare_for_write()
        serialized_sample, shape = self.serialize_sample(sample, break_into_tiles=False)
        self.check_shape_for_update(shape)
        new_nb = (
            None if self.byte_positions_encoder.is_empty() else len(serialized_sample)
        )

        old_data = self._data_bytes
        self._data_bytes = self.create_updated_data(
            local_index, old_data, serialized_sample
        )
        self.update_in_meta_and_headers(local_index, new_nb, shape)
