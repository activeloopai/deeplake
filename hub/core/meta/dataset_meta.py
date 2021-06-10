import hub
import json
from typing import Any, Callable, List

from hub.core.typing import StorageProvider
from hub.util.keys import get_dataset_meta_key


class CallbackList(list):
    def __init__(self, write: Callable):
        self.write = write
        super().__init__()

    def append(self, *args):
        # TODO: only support list/dictionary objects (and parse them to be CallbackDicts/CallbackLists)
        super().append(*args)
        self.write()


class CallbackDict(dict):
    def __init__(self, write: Callable):
        self.write = write
        super().__init__()

    def __setitem__(self, *args):
        # TODO: only support list/dictionary objects (and parse them to be CallbackDicts/CallbackLists)
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

    def asdict(self):
        return {"tensors": self.tensors, "version": self._version, "custom_meta": self.custom_meta}

    def _write(self):
        self.storage[self.key] = bytes(json.dumps(self.asdict()), "utf8")

    def _read(self):
        meta = json.loads(self.storage[self.key])
        self.tensors = meta["tensors"]
        self._version = meta["version"]
        self.custom_meta = meta["custom_meta"]