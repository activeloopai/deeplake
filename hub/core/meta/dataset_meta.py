from typing import Dict, List
from hub.core.storage.provider import StorageProvider
from hub.core.meta.meta import Meta
from hub.util.keys import get_dataset_meta_key


class DatasetMeta(Meta):
    def __init__(self):
        self.tensors = []

        super().__init__()

    def as_dict(self) -> dict:
        d = super().as_dict()
        d["tensors"] = self.tensors
        return d
