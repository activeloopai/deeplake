from hub.util.keys import get_index_meta_key
from hub.core.storage.provider import StorageProvider
from hub.core.meta.meta import CallbackList, Meta


class IndexMeta(Meta):
    @staticmethod
    def create(key: str, storage: StorageProvider):
        # TODO: check if already exists

        required_meta = {"entries": CallbackList}
        return IndexMeta(get_index_meta_key(key), storage, required_meta, allow_custom_meta=False)

    @staticmethod
    def load(key: str, storage: StorageProvider):
        # TODO: check if doesn't exist

        return IndexMeta(get_index_meta_key(key), storage)

    def add_entry(self, entry: dict):
        # TODO: validate entry
        self.entries.append(entry)