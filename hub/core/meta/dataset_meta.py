from typing import Dict, List
from hub.core.storage.provider import StorageProvider
from hub.core.meta.meta import Meta
from hub.util.keys import get_dataset_meta_key


class DatasetMeta(Meta):
    def __init__(self, dataset_meta_key: str, storage: StorageProvider):
        self._tensors = []

        super().__init__(dataset_meta_key, storage)

    @property
    def tensors(self):
        return tuple(self._tensors)

    def add_tensor(self, tensor_name: str):
        self._check_readonly()
        self._tensors.append(tensor_name)
        self.write()

    def write(self):
        super().write(tensors=self._tensors)
