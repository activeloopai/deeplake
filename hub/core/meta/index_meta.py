from hub.util.exceptions import MetaInvalidInitFunctionCall
from hub.util.callbacks import CallbackList
from hub.util.keys import get_index_meta_key
from hub.core.storage.provider import StorageProvider
from hub.core.meta.meta import Meta


class IndexMeta(Meta):
    def __init__(self, *args, **kwargs):  # unused args for faster + better error message
        raise MetaInvalidInitFunctionCall()

    @staticmethod
    def create(key: str, storage: StorageProvider):
        required_meta = {"entries": CallbackList}
        return Meta(
            get_index_meta_key(key), storage, required_meta, allow_custom_meta=False
        )

    @staticmethod
    def load(key: str, storage: StorageProvider):
        return Meta(get_index_meta_key(key), storage)

    def add_entry(self, entry: dict):
        # TODO: validate entry
        self.entries.append(entry)
