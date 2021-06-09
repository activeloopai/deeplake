import json
from typing import List, Tuple, Optional

from numpy import byte

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

    def tobytes(self):
        return self._dict


def _read(payloadBytes: bytearray, key: str, storage: StorageProvider):
    payload = []
    index_map = IndexMap(get_index_map_key(key), storage)
    payload = json.loads(payloadBytes)
    if payload:
        for entry in payload:
            index_map.create_entry(
                chunk_names=entry["chunk_names"],
                start_byte=entry["start_byte"],
                end_byte=entry["end_byte"],
                shape=entry["shape"],
            )

    print(index_map[0].chunk_names())
    return index_map


class IndexMap:
    def __init__(self, key: str, storage: StorageProvider):
        """Initialize a new IndexMap.
        Args:
            key (str): Key for where the index_map is located in `storage` relative to it's root.
            storage (StorageProvider): The storage provider used to access
                the data stored by this dataset.
        """
        self.key: str = get_index_map_key(key)
        self.storage = storage
        if storage.get(self.key) is not None:
            self.state: IndexMap = _read(storage.get(self.key), self.key, self.storage)
        else:
            self.state: list = []

    def add_entry(self, entry: IndexMapEntry):
        """Appends the index map entry to the index map.

        Args:
            entry (IndexMapEntry): The IndexMapEntry object to be stored in the index map
        """
        self.state.append(entry)
        self._write()

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

    def _write(self):
        payloadBytes = bytearray
        payload = [entry.tobytes() for entry in self.state]
        payloadBytes = bytes(json.dumps(payload), "utf-8")
        self.storage[self.key] = payloadBytes

    # def _read(self, storage, index_map):
    #     # print(storage.get(self.key, []))
    #     payloadBytes = bytearray
    #     payload = []
    #     if storage.get(self.key, []) != []:
    #         payloadBytes = storage.get(self.key, [])
    #         payload = json.loads(payloadBytes)
    #     if payload:
    #         for entry in payload:
    #             index_map.create_entry(
    #                 chunk_names=entry["chunk_names"],
    #                 start_byte=entry["start_byte"],
    #                 end_byte=entry["end_byte"],
    #                 shape=entry["shape"],
    #             )

    #     print(index_map[0].chunk_names())
    #     return index_map

    def __len__(self):
        return len(self.state)

    # TODO support slice
    def __getitem__(self, index: int):
        return self.state[index]
