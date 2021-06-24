from hub.core.meta.encode.chunk_name import LAST_INDEX_INDEX
from hub.core.meta.encode.byte_positions import BytePositionsEncoder
from hub.core.meta.encode.shape import ShapeEncoder


class Chunk:
    def __init__(self, min_size_target: int, max_size: int):
        self.min_size_target = min_size_target
        self.max_size = max_size

        self._shape_encoder = ShapeEncoder()
        self._byte_positions_encoder = BytePositionsEncoder()  # TODO

        self._data = bytearray()

    def __len__(self):
        """Returns the length of this chunk's bytes after calling `tobytes`."""
        shape_nbytes = self._shape_encoder.nbytes
        positions_nbytes = self._byte_positions_encoder.nbytes
        return shape_nbytes + 1 + positions_nbytes + 1 + len(self._data)

    def tobytes(self):
        shape_bytes = self._shape_encoder.tobytes()
        positions_bytes = self._byte_positions_encoder.tobytes()
        # TODO: this may copy a lot of bytes
        return shape_bytes + b"\x00" + positions_bytes + b"\x00" + self._data

    def get_sample_bytes(self, sample_index: int):
        sb, eb = self._byte_positions_encoder.get_byte_position(sample_index)
        return self._data[sb, eb]

    def get_sample_shape(self, sample_index: int):
        return self._shape_encoder[sample_index]

    @staticmethod
    def frombuffer(self):
        raise NotImplementedError()

    @property
    def num_data_bytes(self):
        return len(self._data)

    @property
    def has_space(self):
        return self.num_data_bytes < self.min_size_target

    def extend(self, buffer: bytes):
        # TODO: encode start byte / end byte
        if not self.has_space:
            # TODO: exceptions.py
            raise Exception("This chunk does not have space left.")

        buffer_length = len(buffer)

        if buffer_length + len(self._data) > self.max_size:
            raise Exception("Buffer overflow")  # TODO: exceptions.py

        self._data.extend(buffer)

    def __str__(self):
        return f"Chunk(nbytes={len(self._data)})"

    def __repr__(self):
        return str(self)
