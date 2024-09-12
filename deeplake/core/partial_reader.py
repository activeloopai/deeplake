from typing import Dict, Tuple


class PartialReader:
    def __init__(self, cache, path: str, header_offset: int):
        self.cache = cache
        self.path = path
        self.data_fetched: Dict[Tuple[int, int], memoryview] = {}
        self.header_offset = header_offset

    def __getitem__(self, slice_: slice) -> memoryview:
        start = slice_.start + self.header_offset
        stop = slice_.stop + self.header_offset
        step = slice_.step
        assert start is not None and stop is not None
        assert step is None or step == 1
        slice_tuple = (start, stop)

        if start == stop:
            return memoryview(b"")
        if slice_tuple not in self.data_fetched:
            self.data_fetched[slice_tuple] = memoryview(
                self.cache.get_bytes(self.path, start, stop)
            )
        return self.data_fetched[slice_tuple]

    def get_all_bytes(self) -> bytes:
        return self.cache.next_storage[self.path]
