from .storage_factory import StorageFactory as _StorageFactory
from .storage import Storage

amazon = _StorageFactory.amazon
filesystem = _StorageFactory.filesystem
recursive = _StorageFactory.recursive
