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

    return None  # TODO
