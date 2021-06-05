import json
from typing import List, Tuple, Optional

from hub.core.typing import StorageProvider
from hub.util.keys import get_index_map_key
from hub.util.exceptions import InvalidIndexMapEntry


class IndexMapEntry:
    def __init__(
        self,
        chunk_names: Tuple[str],
        start_byte: int,
        end_byte: int,
        shape: Optional[Tuple[int]] = None,
    ):
        """Initialize a new IndexMapEntry.

        Args:
            chunk_names (Tuple[str]): The chunk_names to be stored in the index map entry
            start_byte (int): The start byte of the chunk
            end_byte (int): The end byte of the chunk
            shape (Tuple[int], Optional): The shape of index map entry

        Raises:
            InvalidIndexMapEntry: If an invalid entry is given
        """

        if start_byte < 0:
            raise InvalidIndexMapEntry(start_byte, "start_byte")
        if end_byte < 0:
            raise InvalidIndexMapEntry(end_byte, "end_byte")
        if shape != None:
            for i in range(len(shape)):
                if shape[i] < 0:
                    raise InvalidIndexMapEntry(shape, "shape")

        self._dict = {
            "chunk_names": chunk_names,
            "start_byte": start_byte,
            "end_byte": end_byte,
            "shape": shape,
        }

    def asdict(self):
        return self._dict

    @property
    def chunk_names(self):
        return self._dict["chunk_names"]

    @property
    def start_byte(self):
        return self._dict["start_byte"]

    @property
    def end_byte(self):
        return self._dict["end_byte"]

    @property
    def shape(self):
        return self._dict.get("shape", None)

    # def tobytes(self):
    #     return json.dumps(self._dict)


class IndexMap:
    def __init__(self, key: str, storage: StorageProvider):
        """Initialize a new IndexMap.
        Args:
            key (str): Key for where the index_map is located in `storage` relative to it's root.
            storage (StorageProvider): The storage provider used to access
                the data stored by this dataset.
        """
        self.key: str = get_index_map_key(key)
        self.state: list = storage.get(self.key, [])
        self.storage = storage

    def add_entry(self, entry: IndexMapEntry):
        """Appends the index map entry to the index map.

        Args:
            entry (IndexMapEntry): The IndexMapEntry object to be stored in the index map
        """
        self.state.append(entry)
        self._sync_storage()

    def create_entry(self, **kwargs):
        """Initialize a new IndexMapEntry and calls __add_entry__().

        Args:
            **kwargs: Optional; chunk_names (Tuple[str]): The chunk_names to be stored in the index map entry
            start_byte (int): The start byte of the chunk
            end_byte (int): The end byte of the chunk
            shape (Tuple[int]): The shape of index map entry

        Raises:
            InvalidIndexMapEntry: If an invalid entry is given
        """
        entry = IndexMapEntry(**kwargs)
        self.add_entry(entry)

    def _sync_storage(self):
        self.storage[self.key] = self.state

    def __len__(self):
        return len(self.state)

    # TODO support slice
    def __getitem__(self, index: int):
        return self.state[index]
