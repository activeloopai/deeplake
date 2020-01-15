from typing import Tuple

from .api.hub_bucket import HubBucket
from .api.hub_array import HubArray
from .api.hub_dataset import HubDataset
from .hub_array_impl import HubArrayImpl
from .storage import Storage
from .hub_array_props import HubArrayProps

class HubBucketImpl(HubBucket):
    _storage: Storage = None

    def array_create(self, name: str, shape: Tuple[int, ...], chunk: Tuple[int, ...], dtype: str, compress: str = 'default', compresslevel: float = 0.5, overwrite: bool = False) -> HubArray:
        props = HubArrayProps()
        props.shape = shape
        props.chunk = chunk
        props.dtype = dtype
        props.compress = compress
        props.compresslevel = compresslevel

        assert len(shape) == len(chunk)
        

    def array_open(self, name: str) -> HubArray:
        return HubArrayImpl(name, self._storage)

    def array_delete(self, name: str):
        return self._storage.delete(name)

    def dataset_create(self, name: str, components: dict[str, str], overwrite: bool = False) -> HubDataset:
        raise NotImplementedError()

    def dataset_open(self, name: str) -> HubDataset:
        raise NotImplementedError()

    def dataset_delete(self, name: str):
        raise NotImplementedError()