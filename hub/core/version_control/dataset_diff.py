from hub.core.storage.hub_memory_object import HubMemoryObject
from hub.core.storage import LRUCache
from hub.util.keys import get_dataset_diff_key


class DatasetDiff(HubMemoryObject):
    def __init__(self) -> None:
        self.is_dirty = False
        self.info_updated = False

    def tobytes(self) -> bytes:
        """Returns bytes representation of the dataset diff

        The format stores the following information in order:
        1. The first byte is a boolean value indicating whether the Dataset info was modified or not.
        """
        return b"".join(
            [
                self.info_updated.to_bytes(1, "big"),  # TODO: add other fields
            ]
        )

    @classmethod
    def frombuffer(cls, data: bytes) -> "DatasetDiff":
        """Creates a DatasetDiff object from bytes"""
        dataset_diff = cls()
        dataset_diff.info_updated = bool(int.from_bytes(data[:1], "big"))
        return dataset_diff

    @property
    def nbytes(self):
        """Returns number of bytes required to store the dataset diff"""
        return 1

    def modify_info(self) -> None:
        """Stores information that the info has changed"""
        self.info_updated = True
        self.is_dirty = True


def load_dataset_diff(dataset):
    storage: LRUCache = dataset.storage
    path = get_dataset_diff_key(dataset.version_state["commit_id"])
    try:
        diff = storage.get_hub_object(path, DatasetDiff)
    except KeyError:
        diff = DatasetDiff()
    storage.register_hub_object(path, diff)
    return diff
