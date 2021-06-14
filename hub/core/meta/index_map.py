from hub.constants import META_ENCODING
import json
from typing import List

from hub.core.typing import StorageProvider
from hub.util.keys import get_index_map_key


def write_index_map(key: str, storage: StorageProvider, index_map: list):
    index_map_key = get_index_map_key(key)
    storage[index_map_key] = json.dumps(index_map).encode(META_ENCODING)


def read_index_map(key: str, storage: StorageProvider) -> List[dict]:
    return json.loads(storage[get_index_map_key(key)])
