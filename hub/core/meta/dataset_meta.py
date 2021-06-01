import pickle  # TODO: NEVER USE PICKLE

from hub.core.typing import StorageProvider
from hub.util.keys import get_dataset_meta_key


def write_dataset_meta(storage: StorageProvider, meta: dict):
    storage[get_dataset_meta_key()] = pickle.dumps(meta)


def read_dataset_meta(storage: StorageProvider) -> dict:
    return pickle.loads(storage[get_dataset_meta_key()])
