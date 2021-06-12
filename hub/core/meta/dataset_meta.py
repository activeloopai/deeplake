from hub.core.storage.provider import StorageProvider
from hub.core.meta.meta import CallbackList, Meta
from hub.util.keys import get_dataset_meta_key


class DatasetMeta(Meta):
    @staticmethod
    def create(storage: StorageProvider):
        required_meta = {
            "tensors": CallbackList,
        }

        return DatasetMeta(get_dataset_meta_key(), storage, required_meta=required_meta)

    @staticmethod
    def load(storage: StorageProvider):
        return DatasetMeta(get_dataset_meta_key(), storage)
