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


# TODO: remove this after abhinav's providers are merged to release/2.0 (this is just copy & pasted from @Abhinav's dev branch)
class MemoryProvider:
    def __init__(self):
        self.mapper = {}
        self.max_bytes = 4096  # TODO

    def __getitem__(self, path, start_byte=None, end_byte=None):
        return self.mapper[path][slice(start_byte, end_byte)]

    def __setitem__(self, path, value):
        self.mapper[path] = value

    def __iter__(self):
        yield from self.mapper.items()

    def __delitem__(self, path):
        del self.mapper[path]

    def __len__(self):
        return len(self.mapper.keys())

    @property
    def used_space(self):
        # TODO: this is a slow operation
        return sum([len(b) for b in self.mapper.values()])

    def has_space(self, num_bytes: int) -> bool:
        space_left = self.max_bytes - self.used_space
        return num_bytes <= space_left
