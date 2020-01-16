from .storage_factory import StorageFactory as _StorageFactory
from .storage import Storage

amazon_s3 = _StorageFactory.amazon_s3
filesystem = _StorageFactory.filesystem
recursive = _StorageFactory.recursive
