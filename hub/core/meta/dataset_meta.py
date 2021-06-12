from hub.util.exceptions import MetaInvalidInitFunctionCall
from hub.util.callbacks import CallbackList
from hub.core.storage.provider import StorageProvider
from hub.core.meta.meta import Meta
from hub.util.keys import get_dataset_meta_key


class DatasetMeta(Meta):
    def __init__(self, *args, **kwargs):  # unused args for faster + better error message
        raise MetaInvalidInitFunctionCall()

    @staticmethod
    def create(storage: StorageProvider):
        required_meta = {
            "tensors": CallbackList,
        }

        return Meta(get_dataset_meta_key(), storage, required_meta=required_meta)

    @staticmethod
    def load(storage: StorageProvider):
        return Meta(get_dataset_meta_key(), storage)
