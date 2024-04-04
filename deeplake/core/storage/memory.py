from typing import Any, Dict
from deeplake.core.storage.lru_cache import _get_nbytes
from deeplake.core.storage.provider import StorageProvider
import os


class MemoryProvider(StorageProvider):
    """Provider class for using the memory."""

    def __init__(self, root: str = ""):
        super().__init__()
        self.dict: Dict[str, Any] = {}
        self.root = root

    def _getitem_impl(
        self,
        path: str,
    ):
        return self.dict[path]

    def _setitem_impl(
        self,
        path: str,
        value: bytes,
    ):
        self.check_readonly()
        self.dict[path] = value

    def _delitem_impl(self, path: str):
        self.check_readonly()
        del self.dict[path]

    def _len_impl(self):
        return len(self.dict)

    def _all_keys_impl(self, refresh: bool = False):
        return set(self.dict.keys())

    def _clear_impl(self, prefix=""):
        """Clears the provider."""
        self.check_readonly()
        if prefix:
            self.dict = {k: v for k, v in self.dict.items() if not k.startswith(prefix)}
        else:
            self.dict = {}

    def __getstate__(self) -> dict:
        """Does NOT save the in memory data in state."""
        return {"root": self.root, "_temp_data": self._temp_data}

    def __setstate__(self, state: dict):
        self.__init__(root=state["root"])  # type: ignore
        self._temp_data = state.get("_temp_data", {})

    def get_object_size(self, key: str) -> int:
        return _get_nbytes(self[key])

    def subdir(self, path: str, read_only: bool = False):
        sd = self.__class__(os.path.join(self.root, path))
        sd.read_only = read_only
        return sd
