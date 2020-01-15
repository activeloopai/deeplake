from typing import Tuple, Optional

from .hub_array import HubArray
from .hub_dataset import HubDataset

class HubBucket():
    def array_create(self, name: str, shape: Tuple[int, ...], chunk: Tuple[int, ...], dtype: str, compress: str = 'default', compresslevel: float = 0.5, overwrite: bool = False) -> HubArray:
        raise NotImplementedError()

    def array_open(self, name: str) -> HubArray:
        raise NotImplementedError()

    def array_delete(self, name: str):
        raise NotImplementedError()

    def dataset_create(self, name: str, components: dict[str, str], overwrite: bool = False) -> HubDataset:
        raise NotImplementedError()

    def dataset_open(self, name: str) -> HubDataset:
        raise NotImplementedError()

    def dataset_delete(self, name: str):
        raise NotImplementedError()