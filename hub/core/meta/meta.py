import json
from abc import ABC
from hub.core.storage.provider import StorageProvider
from typing import Any, Callable, Iterable

import hub



class CallbackList(list):
    def __init__(self, write: Callable, raw_list: list=[]):
        self.write = write

        # TODO: handle recursive
        callback_list = []
        for v in raw_list:
            callback_list.append(_convert_to_callback_classes(v, write))

        super().__init__(callback_list)

    def append(self, *args):
        # TODO: only support list/dictionary objects (and parse them to be CallbackDicts/CallbackLists)
        super().append(*args)
        self.write()


class CallbackDict(dict):
    def __init__(self, write: Callable, raw_dict: dict={}):
        self.write = write

        # TODO: handle recursive
        callback_dict = {}
        for k, v in raw_dict.items():
            callback_dict[k] = _convert_to_callback_classes(v, write)
        
        super().__init__(callback_dict)

    def __setitem__(self, *args):
        # TODO: only support list/dictionary objects (and parse them to be CallbackDicts/CallbackLists)
        super().__setitem__(*args)
        self.write()

def _convert_to_callback_classes(value: Any, callback: Callable):
    # TODO: explain what's going on here

    # TODO: check if value is supported `type` (we should only support what json supports)

    if value in (CallbackList, CallbackDict):
        new_value = value(callback)
    elif isinstance(value, dict):
        new_value = CallbackDict(callback, value)
    elif isinstance(value, list):
        new_value = CallbackList(callback, value)
    else:
        new_value = value

    return new_value


def _convert_from_callback_classes(value: Any):
    # TODO: explain what's going on here

    if isinstance(value, CallbackDict):
        new_value = dict(value)
    elif isinstance(value, CallbackList):
        new_value = list(value)
    else:
        new_value = value

    return new_value


class Meta:
    _initialized: bool = False

    def __init__(self, key: str, storage: StorageProvider, required_meta: dict=None):
        self.key = key
        self.storage = storage

        if self.key in self.storage:
            if required_meta is not None:
                raise Exception()  # TODO: exceptions.py or have `create_meta`/`load_meta` functions

            self._read()

        else:
            if required_meta is None:
                raise Exception()  # TODO: exceptions.py or have `create_meta`/`load_meta` functions

            if "version" in required_meta:
                raise Exception("Version is automatically set.")  # TODO: exceptions.py

            required_meta["version"] = hub.__version__
            required_meta["custom_meta"] = CallbackDict(self._write)

            self.from_dict(required_meta)
            self._write()

        self._initialized = True

    def to_dict(self):
        d = {}
        for key in self._required_keys:
            value = getattr(self, key)
            d[key] = _convert_from_callback_classes(value)
        return d

    def from_dict(self, meta: dict):
        for key, value in meta.items():
            new_value = _convert_to_callback_classes(value, self._write)
            setattr(self, key, new_value)
        self._required_keys = meta.keys()
        return self

    def _write(self):
        self.storage[self.key] = bytes(json.dumps(self.to_dict()), "utf8")

    def _read(self):
        meta = json.loads(self.storage[self.key])
        return self.from_dict(meta)

    def __setattr__(self, *args):
        super().__setattr__(*args)
        if self._initialized:
            # can only call `_write` for subsequent setattrs
            self._write()

    # TODO: __str__ & __repr__
    # TODO: if trying to access an attribute that doesn't exist, raise more comprehensive error (what keys are available?)