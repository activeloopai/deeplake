from hub.core.storage.provider import Provider
import fsspec


class LocalProvider(Provider):
    def __init__(self, path):
        self.d = fsspec.filesystem("file").get_mapper(path, check=False, create=False)

    def __getitem__(self, path, start_byte=None, end_byte=None):
        return self.d[path][slice(start_byte, end_byte)]

    def __setitem__(self, path, value):
        self.d[path] = value

    def __iter__(self):
        yield from self.d.items()

    def __delitem__(self, path):
        del self.d[path]

    def __len__(self):
        return len(self.d.keys)