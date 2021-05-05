from hub.core.storage.provider import Provider


class MemoryProvider(Provider):
    def __init__(self):
        self.mapper = fsspec.filesystem("file").get_mapper(path, check=False, create=False)

    def __getitem__(self, path, start_byte=None, end_byte=None):
        return self.mapper[path][slice(start_byte, end_byte)]

    def __setitem__(self, path, value):
        self.mapper[path] = value

    def __iter__(self):
        yield from self.mapper.items()

    def __delitem__(self, path):
        del self.mapper[path]

    def __len__(self):
        return len(self.mapper.keys)
