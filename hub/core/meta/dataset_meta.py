import hub
import json
from typing import Any, Callable, List

from hub.core.typing import StorageProvider


class CallbackList(list):
    def __init__(self, write: Callable):
        self.write = write
        super().__init__()

    def append(self, *args):
        super().append(*args)
        self.write()


class CallbackDict(dict):
    def __init__(self, write: Callable):
        self.write = write
        super().__init__()

    def __setitem__(self, *args):
        super().__setitem__(*args)
        self.write()


class DatasetMeta:
    def __init__(self, key: str, storage: StorageProvider):
        self.key = key
        self.storage = storage

        if self.key in self.storage:
            self._read()
        else:
            self.tensors = CallbackList(self._write)
            self.custom_meta = CallbackDict(self._write)
            self._version = hub.__version__
            self._write()

    @property
    def version(self):
        return self._version

    def _write(self):
        self.storage[self.key] = {"tensors": self.tensors, "version": self._version, "custom_meta": self.custom_meta}

    def _read(self):
        meta = self.storage[self.key]
        self.tensors = meta["tensors"]
        self._version = meta["version"]
        self.custom_meta = meta["custom_meta"]