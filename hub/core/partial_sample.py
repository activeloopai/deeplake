from typing import Optional, Tuple, Union, List, Any
from hub.core.storage.provider import StorageProvider


class PartialSample(object):
    def __init__(self, storage: StorageProvider, sample_size: Optional[Tuple[int, ...]] = None, tile_size: Optional[Tuple[int, ...]] = None):
        self.storage = storage
        self.sample_size = sample_size
        self.tile_size = tile_size

    def __setitem__(self, index: Union[slice, int, List[int]], item: Any):
        pass

    def __getitem__(self, index: Union[slice, int, List[int]]):
        pass

    