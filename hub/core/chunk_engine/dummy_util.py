class DummySampleCompression:
    @property
    def __name__(self):
        return "dummy_sample"

    @property
    def subject(self):
        return "sample"

    @staticmethod
    def compress(x):
        return x

    @staticmethod
    def decompress(x):
        return x


class DummyChunkCompression:
    @property
    def __name__(self):
        return "dummy_chunk"

    @property
    def subject(self):
        return "chunk"

    @staticmethod
    def compress(x):
        return x

    @staticmethod
    def decompress(x):
        return x


# TODO: obviously this is terrible...
dummy_compression_map = {
    DummySampleCompression().__name__: DummySampleCompression,
    DummyChunkCompression().__name__: DummyChunkCompression,
}
