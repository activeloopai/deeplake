class BytePositionsEncoder:
    _encoded = None

    @property
    def num_samples(self) -> int:
        if self._encoded is None:
            return 0
        return int(self._encoded[-1, -1] + 1)

    def tobytes(self) -> bytes:
        # TODO:
        return bytes()

    @property
    def nbytes(self):
        if self._encoded is None:
            return 0
        return self._encoded.nbytes

    def add_byte_position(self, num_bytes: int, num_samples: int):
        raise NotImplementedError()

    def get_byte_position(self, sample_index: int):
        raise NotImplementedError()
