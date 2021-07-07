from hub.util.exceptions import FullChunkError
import hub
from hub.core.storage.cachable import Cachable
from typing import Sequence, Tuple, Union
import numpy as np
from io import BytesIO

from hub.core.meta.encode.shape import ShapeEncoder
from hub.core.meta.encode.byte_positions import BytePositionsEncoder

from hub.core.lowlevel import encode, decode, malloc, _write_pybytes


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

        self._data: List[memoryview] = [] if data is None else [data]

    @property
    def memoryview_data(self):
        # deprecated
        if len(self._data) == 1:
            return self._data[0]
        ptr = malloc(sum(map(len, self._data)))
        for data in self._data:
            ptr = _write_pybytes(ptr, data)
        return memoryview(ptr.bytes)

    def _get_2d_idx(self, idx):
        i = 0
        while len(self._data[i]) <= idx:
            i += 1
            idx -= len(self._data[i])
        return i, idx

    def view(self, start, end):
        if len(self._data) == 1:
            return self._data[0][start:end]
        start2d = self._get_2d_idx(start)
        end2d = self._get_2d_idx(end)
        byts = []
        byts.append(self._data[start2d[0]][start2d[1] :])
        for i in range(start2d[0] + 1, end2d[0]):
            byts.append(self._data[i])
        byts.append(self._data[end2d[0]][: end2d[1]])
        ptr = malloc(end - start)
        for byt in byts:
            ptr = _write_pybytes(ptr, byt)
        return memoryview(ptr.bytes)

    @property
    def num_data_bytes(self):
        return sum(map(len, self._data))

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

        # `_data` will be a `memoryview` if `frombuffer` is called.
        # if isinstance(self._data, memoryview):
        #     self._data = bytearray(self._data)

        # note: incoming_num_bytes can be 0 (empty sample)
        self._data.append(buffer)
        self.update_headers(incoming_num_bytes, shape)

    def update_headers(self, incoming_num_bytes: int, sample_shape: Tuple[int]):
        """Updates this chunk's header. A chunk should NOT exist without headers.

        Args:
            incoming_num_bytes (int): The length of the buffer that was used to
            sample_shape (Tuple[int]): Every sample that `num_samples` symbolizes is considered to have `sample_shape`.

        Raises:
            ValueError: If `incoming_num_bytes` is not divisible by `num_samples`.
        """

        num_bytes_per_sample = incoming_num_bytes
        self.shapes_encoder.add_shape(sample_shape, 1)
        self.byte_positions_encoder.add_byte_position(num_bytes_per_sample, 1)

    def __len__(self):
        """Calculates the number of bytes `tobytes` will be without having to call `tobytes`. Used by `LRUCache` to determine if this chunk can be cached."""

        shape_nbytes = self.shapes_encoder.nbytes
        range_nbytes = self.byte_positions_encoder.nbytes
        error_bytes = 32  # to account for any extra delimeters/stuff that `np.savez` may create in excess

        return shape_nbytes + range_nbytes + self.num_data_bytes + error_bytes

    def tobytes(self) -> memoryview:
        return encode(
            hub.__version__,
            self.shapes_encoder.array,
            self.byte_positions_encoder.array,
            self._data,
        )

    @classmethod
    def frombuffer(cls, buffer: bytes) -> "Chunk":
        version, shapes, byte_positions, data = decode(buffer)
        return cls(shapes, byte_position, data=data)
