from typing import Dict, List, Tuple
from hub.util.keys import get_index_meta_key
from hub.core.storage.provider import StorageProvider
from hub.core.meta.meta import Meta


def _create_entry(
    chunk_names: List[str],
    start_byte: int,
    end_byte: int,
    shape: Tuple[int],
) -> dict:
    # TODO: replace with `SampleMeta` class

    entry = {
        "chunk_names": chunk_names,
        "start_byte": start_byte,
        "end_byte": end_byte,
        "shape": shape,
    }

    return entry


class IndexMeta(Meta):
    entries: List

    @staticmethod
    def create(key: str, storage: StorageProvider):
        """Index metadata is responsible for keeping track of where chunked samples exist.

        Note:
            Index metadata that is automatically synchronized with `storage`. For more details, see the `Meta` class.
            Auto-populates `required_meta` that `Meta` accepts as an argument.

        Args:
            key (str): Key relative to `storage` where this instance will be synchronized to. Will automatically add the tensor meta filename to the end.
            storage (StorageProvider): Destination of this meta.

        Returns:
            IndexMeta: Index meta object.
        """

        required_meta: Dict = {"entries": []}
        return IndexMeta(
            get_index_meta_key(key), storage, required_meta, allow_custom_meta=False
        )

    @staticmethod
    def load(key: str, storage: StorageProvider):
        return IndexMeta(get_index_meta_key(key), storage)

    def add_entry(
        self,
        chunk_names: List[str],
        start_byte: int,
        end_byte: int,
        shape: Tuple[int],
    ):
        self.entries.append(_create_entry(chunk_names, start_byte, end_byte, shape))
