import pickle

from .write import MemoryProvider

from typing import Callable

# TODO change storage type to StorageProvider
def read(
    key: str,
    index: int,
    storage: MemoryProvider,
    decompressor: Callable,
):
    """
    array <- bytes <- decompressor <- chunks <- storage
    """

    # TODO: don't use pickle
    index_map = pickle.loads(storage["index_map"])
    print(index_map)
    return None  # TODO
