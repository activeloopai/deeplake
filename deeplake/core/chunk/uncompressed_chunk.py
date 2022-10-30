from io import BufferedWriter
import numpy as np
from typing import List, Union
from deeplake.core.serialize import check_sample_shape, bytes_to_text
from deeplake.core.tiling.sample_tiles import SampleTiles
from deeplake.core.polygon import Polygons
from deeplake.util.casting import intelligent_cast
from deeplake.util.exceptions import EmptyTensorError, TensorDtypeMismatchError
from .base_chunk import BaseChunk, InputSample


class UncompressedChunk(BaseChunk):
    def extend_if_has_space(  # type: ignore
        self,
        incoming_samples: Union[List[InputSample], np.ndarray],
        update_tensor_meta: bool = True,
    ) -> float:
        self.prepare_for_write()
        if isinstance(incoming_samples, np.ndarray):
            if incoming_samples.dtype == object:
                incoming_samples = list(incoming_samples)
            else:
                return self._extend_if_has_space_numpy(
                    incoming_samples, update_tensor_meta
                )
        else:
            sample_shape = self.tensor_meta.sample_shape
            if sample_shape and None not in sample_shape:
                return self._extend_if_has_space_numpy(
                    incoming_samples, update_tensor_meta
                )
            # types = set(map(type, incoming_samples))
            # if types == {np.ndarray}:
            #     dtypes = set((s.dtype for s in incoming_samples))
            #     if len(dtypes) == 1:
            #         shapes = set((s.shape for s in incoming_samples))
            #         if len(shapes) == 1:
            #             return self._extend_if_has_space_numpy(
            #                 incoming_samples, update_tensor_meta
            #             )
            # elif all((isinstance(s, (int, float, bool)) or np.isscalar(s) for s in incoming_samples)):
            #     return self._extend_if_has_space_numpy(
            #         np.array(incoming_samples, dtype=self.dtype), update_tensor_meta
            #     )
            # TODO detect multidim uniform sequences
        print(incoming_samples)
        return self._extend_if_has_space_list(incoming_samples, update_tensor_meta)

    def _extend_if_has_space_numpy(
        self,
        incoming_samples: np.ndarray,
        update_tensor_meta: bool = True,
    ) -> float:
        num_samples: int
        elem = incoming_samples[0]
        shape = elem.shape
        size = elem.size
        if not shape:
            shape = (1,)
        self.num_dims = self.num_dims or len(shape)
        if size == 0:
            num_samples = len(incoming_samples)
        else:
            num_data_bytes = self.num_data_bytes
            num_samples = int((self.min_chunk_size - num_data_bytes) / elem.nbytes)
            if num_samples == 0 and num_data_bytes == 0 and self.tiling_threshold < 0:
                num_samples = 1

        samples = incoming_samples[:num_samples]
        my_dtype = self.dtype
        if isinstance(samples, np.ndarray):
            if samples.dtype != my_dtype:
                if not np.can_cast(samples.dtype, my_dtype):
                    raise TensorDtypeMismatchError(
                        my_dtype,
                        samples.dtype,
                        self.htype,
                    )
                samples = samples.astype(my_dtype)
            self.data_bytes += samples.tobytes()
        else:  # list
            dtypes = set((s.dtype for s in samples))
            if dtypes != {my_dtype}:
                for dtype in dtypes:
                    if not np.can_cast(dtype, my_dtype):
                        raise TensorDtypeMismatchError(
                            my_dtype,
                            dtype,
                            self.htype,
                        )
                samples = [s.astype(my_dtype) for s in samples]
            self.data_bytes += b"".join([s.tobytes() for s in samples])

        if num_samples > 0:
            self.register_in_meta_and_headers(
                incoming_samples[0].nbytes,
                shape,
                update_tensor_meta=update_tensor_meta,
                num_samples=num_samples,
            )
        return float(num_samples)

    def _extend_if_has_space_list(
        self,
        incoming_samples: List[InputSample],
        update_tensor_meta: bool = True,
    ) -> float:
        raise Exception()
        num_samples: float = 0
        for i, incoming_sample in enumerate(incoming_samples):
            serialized_sample, shape = self.serialize_sample(incoming_sample)
            if shape is not None:
                self.num_dims = self.num_dims or len(shape)
                check_sample_shape(shape, self.num_dims)

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
                    self.data_bytes += serialized_sample  # type: ignore

                    self.register_in_meta_and_headers(
                        sample_nbytes,
                        shape,
                        update_tensor_meta=update_tensor_meta,
                    )
                    num_samples += 1
                else:
                    break

        return num_samples

    def read_sample(
        self,
        local_index: int,
        cast: bool = True,
        copy: bool = False,
        decompress: bool = True,
        is_tile: bool = False,
    ):
        if self.is_empty_tensor:
            raise EmptyTensorError(
                "This tensor has only been populated with empty samples. "
                "Need to add at least one non-empty sample before retrieving data."
            )
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
            shape = self.shapes_encoder[local_index]
            if not bps.is_empty():
                sb, eb = bps[local_index]
                buffer = buffer[sb:eb]

        if not decompress:
            if copy:
                buffer = bytes(buffer)
            return buffer
        if self.is_text_like:
            buffer = bytes(buffer)
            return bytes_to_text(buffer, self.htype)
        if self.tensor_meta.htype == "polygon":
            return Polygons.frombuffer(
                buffer,
                dtype=self.tensor_meta.dtype,
                ndim=shape[-1],
            )
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

        old_data = self.data_bytes
        self.data_bytes = self.create_updated_data(
            local_index, old_data, serialized_sample
        )
        self.update_in_meta_and_headers(local_index, new_nb, shape)
