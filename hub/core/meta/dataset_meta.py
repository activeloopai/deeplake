from hub.core.meta.meta import CallbackDict, CallbackList, Meta
from hub.core.typing import StorageProvider


def create_dataset_meta(key: str, storage: StorageProvider) -> Meta:
    required_meta = {
        "tensors": CallbackList,
    }

    return Meta(key, storage, required_meta)

def load_dataset_meta(key: str, storage: StorageProvider) -> Meta:
    return Meta(key, storage)