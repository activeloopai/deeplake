from hub.core.storage.provider import Provider
import fsspec


class S3AIOBotoProvider(Provider):
    def __init__(self, path):
        self.d = fsspec.filesystem("s3").get_mapper(path, check=False, create=False)

    def __getitem__(self, path):
        return self.d[path]

    def __setitem__(self, path, value):
        self.d[path] = value

    def __iter__(self):
        yield from self.d.items()

    def __delitem__(self, path):
        del self.d[path]

    def __len__(self):
        return len(self.d.keys)
