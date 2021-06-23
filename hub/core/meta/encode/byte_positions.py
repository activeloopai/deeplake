class BytePositionsEncoder:
    _encoded = None

    def add_byte_position(self, num_bytes: int, num_samples: int):
        raise NotImplementedError()
