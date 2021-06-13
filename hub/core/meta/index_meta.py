from typing import List, Tuple
from hub.util.exceptions import MetaInvalidInitFunctionCall
from hub.util.callbacks import CallbackDict, CallbackList
from hub.util.keys import get_index_meta_key
from hub.core.storage.provider import StorageProvider
from hub.core.meta.meta import Meta


def _create_entry(chunk_names: List[str], start_byte: int, end_byte: int, shape: Tuple[int]=None) -> dict:
    entry = {
        "chunk_names": chunk_names,
        "start_byte": start_byte,
        "end_byte": end_byte,
    }
    if shape is not None:
        entry["shape"] = shape
    return entry


class IndexMeta(Meta):
    @staticmethod
    def create(key: str, storage: StorageProvider):
        required_meta = {"entries": CallbackList}
        return IndexMeta(
            get_index_meta_key(key), storage, required_meta, allow_custom_meta=False
        )

    @staticmethod
    def load(key: str, storage: StorageProvider):
        return IndexMeta(get_index_meta_key(key), storage)

    def add_entry(self, chunk_names: List[str], start_byte: int, end_byte: int, shape: Tuple[int]=None):
        self.entries.append(_create_entry(chunk_names, start_byte, end_byte, shape))
