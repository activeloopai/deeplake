import pickle
from typing import List

from hub.core.typing import StorageProvider
from hub.util.keys import get_index_map_key


def write_index_map(key: str, storage: StorageProvider, index_map: list):
    _validate_index_map(index_map)
    index_map_key = get_index_map_key(key)
    storage[index_map_key] = pickle.dumps(index_map)

def read_index_map(key: str, storage: StorageProvider) -> List[dict]:
    return pickle.loads(storage[get_index_map_key(key)])

def _validate_index_map(index_map: list):
    # TODO
    pass