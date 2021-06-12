from hub.constants import META_ENCODING
import json

from hub.core.typing import StorageProvider
from hub.util.keys import get_dataset_meta_key


def write_dataset_meta(storage: StorageProvider, meta: dict):
    storage[get_dataset_meta_key()] = json.dumps(meta).encode(META_ENCODING)


def read_dataset_meta(storage: StorageProvider) -> dict:
    return json.loads(storage[get_dataset_meta_key()])
