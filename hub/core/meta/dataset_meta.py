from hub.core.meta.meta import CallbackDict, CallbackList, Meta
import hub

from hub.core.typing import StorageProvider
from hub.util.keys import get_dataset_meta_key


def create_dataset_meta(key: str, storage: StorageProvider):
    required_meta = {
        "tensors": CallbackList,
        "custom_meta": CallbackDict
    }

    return Meta(key, storage, required_meta)

def load_dataset_meta(key: str, storage: StorageProvider):
    return Meta(key, storage)

#class DatasetMeta(Meta):
#    def __init__(self, **kwargs):
#        required_meta = {
#            "tensors": CallbackList(self._write),
#            "custom_meta": CallbackDict(self._write)
#        }
#
#        print(kwargs)
#        super().__init__(required_meta=required_meta, **kwargs)