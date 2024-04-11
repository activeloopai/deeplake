from deeplake.core.storage.provider import StorageProvider
from deeplake.core.partial_reader import PartialReader
from deeplake.core.storage.deeplake_memory_object import DeepLakeMemoryObject
from typing import Optional, Union, Dict


class IndraProvider(StorageProvider):
    """Provider class for using Indra storage provider."""

    def __init__(
        self,
        root,  # Union[str, storage.provider],
        read_only: Optional[bool] = False,
        **kwargs,
    ):
        from indra.api import storage  # type: ignore

        if isinstance(root, str):
            self.core = storage.create(root, read_only, **kwargs)
        else:
            self.core = root
        self.root = self.path

    @property
    def path(self):
        return self.core.path

    @property
    def original_path(self):
        return self.core.original_path

    @property
    def token(self):
        return self.core.token

    def copy(self):
        return IndraProvider(self.core)

    def subdir(self, path: str, read_only: bool = False):
        return IndraProvider(self.core.subdir(path, read_only))

    def __setitem__(self, path, content):
        self.check_readonly()
        self.core.set(path, bytes(content))

    def __getitem__(self, path):
        return bytes(self.core.get(path))

    def get_bytes(
        self, path, start_byte: Optional[int] = None, end_byte: Optional[int] = None
    ):
        s = start_byte or 0
        e = end_byte or 0
        try:
            return bytes(self.core.get(path, s, e))
        except RuntimeError as e:
            raise KeyError(path)

    def get_deeplake_object(
        self,
        path: str,
        expected_class,
        meta: Optional[Dict] = None,
        url=False,
        partial_bytes: int = 0,
    ):
        if partial_bytes != 0:
            buff = self.get_bytes(path, 0, partial_bytes)
            obj = expected_class.frombuffer(buff, meta, partial=True)
            obj.data_bytes = PartialReader(self, path, header_offset=obj.header_bytes)
            return obj

        item = self[path]
        if isinstance(item, DeepLakeMemoryObject):
            if type(item) != expected_class:
                raise ValueError(
                    f"'{path}' was expected to have the class '{expected_class.__name__}'. Instead, got: '{type(item)}'."
                )
            return item

        if isinstance(item, (bytes, memoryview)):
            obj = (
                expected_class.frombuffer(item)
                if meta is None
                else expected_class.frombuffer(item, meta)
            )
            return obj

        raise ValueError(f"Item at '{path}' got an invalid type: '{type(item)}'.")

    def get_object_size(self, path: str) -> int:
        return self.core.length(path)

    def __delitem__(self, path):
        return self.core.remove(path)

    def _all_keys(self):
        return self.core.list("")

    def __len__(self):
        return len(self.core.list(""))

    def __iter__(self):
        return iter(self.core.list(""))

    def clear(self, prefix=""):
        self.core.clear(prefix)
