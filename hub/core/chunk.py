from hub.util.exceptions import FullChunkError, TensorInvalidSampleShapeError
import hub
from hub.core.storage.cachable import Cachable
from typing import Tuple, Union
import numpy as np

from hub.core.meta.encode.shape import ShapeEncoder
from hub.core.meta.encode.byte_positions import BytePositionsEncoder

from hub.core.serialize import serialize_chunk, deserialize_chunk, infer_chunk_num_bytes


class Chunk(Cachable):
    def __init__(
        self,
        encoded_shapes: np.ndarray = None,
        encoded_byte_positions: np.ndarray = None,
        data: memoryview = None,
    ):
        """Blob storage of bytes. Tensor data is split into chunks of roughly the same size.
        `ChunkEngine` handles the creation of `Chunk`s and the delegation of samples to them.

        Data layout:
            Every chunk has data and a header.

            Header:
                All samples this chunk contains need 2 components: shape and byte position.
                `ShapeEncoder` handles encoding the `start_byte` and `end_byte` for each sample.
                `BytePositionsEncoder` handles encoding the `shape` for each sample.

            Data:
                All samples this chunk contains are added into `_data` in bytes form directly adjacent to one another, without
                delimeters.

            See `tobytes` and `frombytes` for more on how chunks are serialized

        Args:
            encoded_shapes (np.ndarray): Used to construct `ShapeEncoder` if this chunk already exists. Defaults to None.
            encoded_byte_positions (np.ndarray): Used to construct `BytePositionsEncoder` if this chunk already exists.
                Used by `frombuffer`. Defaults to None.
            data (memoryview): If this chunk already exists, data should be set.
                Used by `frombuffer`. Defaults to None.
        """

        self.shapes_encoder = ShapeEncoder(encoded_shapes)
        self.byte_positions_encoder = BytePositionsEncoder(encoded_byte_positions)

        self._data: Union[memoryview, bytearray] = data or bytearray()

    @property
    def memoryview_data(self):
        if isinstance(self._data, memoryview):
            return self._data
        return memoryview(self._data)

    def _make_data_bytearray(self):
        """Copies `self._data` into a bytearray if it is a memoryview."""

        # `_data` will be a `memoryview` if `frombuffer` is called.
        if isinstance(self._data, memoryview):
            self._data = bytearray(self._data)

    @property
    def num_data_bytes(self):
        return len(self._data)

    def is_under_min_space(self, min_data_bytes_target: int) -> bool:
        """If this chunk's data is less than `min_data_bytes_target`, returns True."""

        return self.num_data_bytes < min_data_bytes_target

    def has_space_for(self, num_bytes: int, max_data_bytes: int):
        return self.num_data_bytes + num_bytes <= max_data_bytes

    def append_sample(self, buffer: memoryview, max_data_bytes: int, shape: Tuple[int]):
        """Store `buffer` in this chunk.

        Args:
            buffer (memoryview): Buffer that represents a single sample.
            max_data_bytes (int): Used to determine if this chunk has space for `buffer`.
            shape (Tuple[int]): Shape for the sample that `buffer` represents.

        Raises:
            FullChunkError: If `buffer` is too large.
        """

        incoming_num_bytes = len(buffer)

        if not self.has_space_for(incoming_num_bytes, max_data_bytes):
            raise FullChunkError(
                f"Chunk does not have space for the incoming bytes (incoming={incoming_num_bytes}, max={max_data_bytes})."
            )

        self._make_data_bytearray()

        # note: incoming_num_bytes can be 0 (empty sample)
        self._data += buffer  # type: ignore
        self.register_sample_to_headers(incoming_num_bytes, shape)

    def register_sample_to_headers(
        self, incoming_num_bytes: int, sample_shape: Tuple[int]
    ):
        """Registers a single sample to this chunk's header. A chunk should NOT exist without headers.

        Args:
            incoming_num_bytes (int): The length of the buffer that was used to
            sample_shape (Tuple[int]): Every sample that `num_samples` symbolizes is considered to have `sample_shape`.

        Raises:
            ValueError: If `incoming_num_bytes` is not divisible by `num_samples`.
        """

        num_bytes_per_sample = incoming_num_bytes
        self.shapes_encoder.register_samples(sample_shape, 1)
        self.byte_positions_encoder.register_samples(num_bytes_per_sample, 1)

    def update_sample(
        self, local_sample_index: int, new_buffer: memoryview, new_shape: Tuple[int]
    ):
        """Updates data and headers for `local_sample_index` with the incoming `new_buffer` and `new_shape`."""

        expected_dimensionality = len(self.shapes_encoder[local_sample_index])
        if expected_dimensionality != len(new_shape):
            raise TensorInvalidSampleShapeError(new_shape, expected_dimensionality)

        new_nb = len(new_buffer)

        # get the unchanged data
        old_start_byte, old_end_byte = self.byte_positions_encoder[local_sample_index]
        left = self._data[:old_start_byte]
        right = self._data[old_end_byte:]

        # update encoders
        self.byte_positions_encoder[local_sample_index] = new_nb
        self.shapes_encoder[local_sample_index] = new_shape
        new_start_byte, new_end_byte = self.byte_positions_encoder[local_sample_index]

        # preallocate
        total_new_bytes = len(left) + new_nb + len(right)
        new_data = bytearray(total_new_bytes)

        # copy old data and add new data
        new_data[:new_start_byte] = left
        new_data[new_start_byte:new_end_byte] = new_buffer
        new_data[new_end_byte:] = right
        self._data = new_data

    @property
    def nbytes(self):
        """Calculates the number of bytes `tobytes` will be without having to call `tobytes`. Used by `LRUCache` to determine if this chunk can be cached."""

        return infer_chunk_num_bytes(
            hub.__version__,
            self.shapes_encoder.array,
            self.byte_positions_encoder.array,
            len_data=len(self._data),
        )

    def tobytes(self) -> memoryview:
        return serialize_chunk(
            hub.__version__,
            self.shapes_encoder.array,
            self.byte_positions_encoder.array,
            [self._data],
        )

    @classmethod
    def frombuffer(cls, buffer: bytes):
        if not buffer:
            return cls()
        version, shapes, byte_positions, data = deserialize_chunk(buffer)
        return cls(shapes, byte_positions, data=data)
