from typing import List, Tuple
from hub.util.callbacks import CallbackList
from hub.util.keys import get_index_meta_key
from hub.core.storage.provider import StorageProvider
from hub.core.meta.meta import Meta


def _create_entry(
    chunk_names: List[str], start_byte: int, end_byte: int, shape: Tuple[int] = None
) -> dict:
    # TODO: replace with `SampleMeta` class

    entry = {
        "chunk_names": chunk_names,
        "start_byte": start_byte,
        "end_byte": end_byte,
    }
    if shape is not None:
        entry["shape"] = shape
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
            htype (str): All tensors require an `htype`. This determines the default meta keys/values.
            **kwargs: Any key that the provided `htype` has can be overridden via **kwargs. For more information, check out `hub.htypes`.

        Raises:
            TensorMetaInvalidHtypeOverwriteKey: If **kwargs contains unsupported keys for the provided `htype`.
            TensorMetaInvalidHtypeOverwriteValue: If **kwargs contains unsupported values for the keys of the provided `htype`.
        """

        required_meta = {"entries": []}
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
        shape: Tuple[int] = None,
    ):
        self.entries.append(_create_entry(chunk_names, start_byte, end_byte, shape))
