from hub.util.exceptions import FullChunkError
import hub
from hub.core.storage.cachable import Cachable
from typing import Sequence, Tuple, Union
import numpy as np
from io import BytesIO

from hub.core.meta.encode.shape import ShapeEncoder
from hub.core.meta.encode.byte_positions import BytePositionsEncoder


class Chunk(Cachable):
    def __init__(
        self,
        encoded_shapes: np.ndarray = None,
        encoded_byte_positions: np.ndarray = None,
        data: memoryview = None,
    ):
        """Blob storage of bytes."""

        self.shapes_encoder = ShapeEncoder(encoded_shapes)
        self.byte_positions_encoder = BytePositionsEncoder(encoded_byte_positions)

        self._data: Union[bytearray, memoryview] = data or bytearray()

    @property
    def memoryview_data(self):
        return memoryview(self._data)

    @property
    def num_data_bytes(self):
        return len(self._data)

    def is_under_min_space(self, min_data_bytes_target: int):
        return self.num_data_bytes < min_data_bytes_target

    def has_space_for(self, num_bytes: int, max_data_bytes: int):
        return self.num_data_bytes + num_bytes <= max_data_bytes

    def append(
        self, incoming_buffer: memoryview, max_data_bytes: int
    ) -> Tuple["Chunk"]:
        """Store `incoming_buffer` in this chunk.

        Raises:
            FullChunkError: If `incoming_buffer` is too large.
        """

        incoming_num_bytes = len(incoming_buffer)

        if not self.has_space_for(incoming_num_bytes, max_data_bytes):
            raise FullChunkError(
                f"Chunk does not have space for the incoming bytes (incoming={incoming_num_bytes}, max={max_data_bytes})."
            )

        # note: incoming_num_bytes can be 0 (empty sample)
        self._data += incoming_buffer

    def update_headers(
        self, incoming_num_bytes: int, num_samples: int, sample_shape: Sequence[int]
    ):
        """Updates this chunk's header. A chunk may exist without headers, that is up to the `ChunkEngine` to delegate.

        Args:
            incoming_num_bytes (int): Number of bytes this header should account for. Should be divisble by `num_samples`.
            num_samples (int): Number of samples this header should account for.
            sample_shape (Sequence[int]): Every sample that `num_samples` symbolizes is considered to have `sample_shape`.

        Raises:
            Exception: If trying to update headers when no data was actually added.
            ValueError: If `incoming_num_bytes` is not divisible by `num_samples`.
        """

        if incoming_num_bytes % num_samples != 0:
            raise ValueError(
                "Incoming bytes should be divisible by the number of samples to properly update headers."
            )

        num_bytes_per_sample = incoming_num_bytes // num_samples
        self.shapes_encoder.add_shape(sample_shape, num_samples)
        self.byte_positions_encoder.add_byte_position(num_bytes_per_sample, num_samples)

    def __len__(self):
        # this should not call `tobytes` because it will be slow. should calculate the amount of bytes this chunk takes up in total. (including headers)

        shape_nbytes = self.shapes_encoder.nbytes
        range_nbytes = self.byte_positions_encoder.nbytes
        error_bytes = 32  # to account for any extra delimeters/stuff that `np.savez` may create in excess

        return shape_nbytes + range_nbytes + self.num_data_bytes + error_bytes

    def tobytes(self) -> memoryview:
        out = BytesIO()

        # TODO: for fault tolerance, we should have a chunk store the ID for the next chunk
        # TODO: in case the index chunk meta gets pwned (especially during a potentially failed transform job merge)

        np.savez(
            out,
            version=hub.__encoded_version__,
            shapes=self.shapes_encoder.array,
            byte_positions=self.byte_positions_encoder.array,
            data=np.frombuffer(self.memoryview_data, dtype=np.uint8),
        )
        out.seek(0)
        return out.getbuffer()

    @classmethod
    def frombuffer(cls, buffer: bytes):
        bio = BytesIO(buffer)
        npz = np.load(bio)

        return cls(npz["shapes"], npz["byte_positions"], data=npz["data"].tobytes())
