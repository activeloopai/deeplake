from deeplake.core.storage.provider import StorageProvider
from indra.api import storage  # type: ignore
from typing import Optional, Union


class IndraProvider(StorageProvider):
    """Provider class for using Indra storage provider."""

    def __init__(
        self,
        root: Union[str, storage.provider],
        read_only: Optional[bool] = False,
        **kwargs,
    ):
        if isinstance(root, str):
            self.core = storage.create(root, read_only, **kwargs)
        else:
            self.core = root

    def copy(self):
        return IndraProvider(self.core)

    def subdir(self, path: str, read_only: bool = False):
        return IndraProvider(self.core.subdir(path, read_only))

    def __setitem__(self, path, content):
        self.check_readonly()
        self.core.set(path, content)

    def __getitem__(self, path):
        try:
            return bytes(self.core.get(path))
        except RuntimeError as e:
            raise KeyError(path)

    def get_bytes(
        self, path, start_byte: Optional[int] = None, end_byte: Optional[int] = None
    ):
        s = start_byte or 0
        e = end_byte or 0
        try:
            return bytes(self.core.get(path, s, e))
        except RuntimeError as e:
            raise KeyError(path)

    def get_object_size(self, path: str) -> int:
        try:
            return self.core.length(path)
        except RuntimeError as e:
            raise KeyError(path)

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
