from typing import Optional
from hub.core.storage.provider import StorageProvider


class PartialReader:
    def __init__(self, storage_provider: StorageProvider, path: str):
        self.storage_provider = storage_provider
        self.path = path
        self.data_fetched = {}

    def __getitem__(self, slice_: slice):
        if self.data_bytes is not None:
            return self.data_bytes[slice_]
        start = slice_.start
        stop = slice_.stop
        step = slice_.step
        assert start is not None and stop is not None
        assert step is None or step == 1
        if slice_ not in self.data_fetched:
            self.data_fetched[slice_] = self.storage_provider.get_bytes(
                self.path, start, stop
            )
        return self.data_fetched[slice_]

    def __len__(self):
        return sum(slice_.stop - slice_.start for slice_ in self.data_fetched)

    def get_all_bytes(self) -> bytes:
        return self.storage_provider[self.path]
