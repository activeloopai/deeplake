from typing import Dict, Optional
from hub.core.storage.provider import StorageProvider


class PartialReader:
    def __init__(self, storage_provider: StorageProvider, path: str):
        self.storage_provider = storage_provider
        self.path = path
        self.data_fetched: Dict[tuple[int, int], bytes] = {}

    def __getitem__(self, slice_: slice):
        start = slice_.start
        stop = slice_.stop
        step = slice_.step
        assert start is not None and stop is not None
        assert step is None or step == 1
        slice_tuple = (start, stop)
        if slice_tuple not in self.data_fetched:
            self.data_fetched[slice_tuple] = self.storage_provider.get_bytes(
                self.path, start, stop
            )
        return self.data_fetched[slice_tuple]

    def __len__(self):
        return sum(stop - start for start, stop in self.data_fetched)

    def get_all_bytes(self) -> bytes:
        return self.storage_provider[self.path]
