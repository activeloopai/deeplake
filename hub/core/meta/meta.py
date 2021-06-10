import json
from abc import ABC
from hub.core.storage.provider import StorageProvider
from typing import Any, Callable

import hub



class CallbackList(list):
    def __init__(self, write: Callable, *args):
        self.write = write
        super().__init__(*args)

    def append(self, *args):
        # TODO: only support list/dictionary objects (and parse them to be CallbackDicts/CallbackLists)
        super().append(*args)
        self.write()


class CallbackDict(dict):
    def __init__(self, write: Callable, *args):
        self.write = write
        super().__init__(*args)

    def __setitem__(self, *args):
        # TODO: only support list/dictionary objects (and parse them to be CallbackDicts/CallbackLists)
        super().__setitem__(*args)
        print("setitem", args)
        self.write()

def _convert_to_callback_classes_recursively(value: Any, callback: Callable):
    # TODO: explain what's going on here

    if value in (CallbackList, CallbackDict):
        new_value = value(callback)
    elif isinstance(value, dict):
        new_value = CallbackDict(callback, value)
    elif isinstance(value, list):
        new_value = CallbackList(callback, value)
    else:
        new_value = value

    # TODO: recursive (handle nested dicts and lists)

    return new_value


def _convert_from_callback_classes_recursively(value: Any):
    if isinstance(value, CallbackDict):
        new_value = dict(value)
    elif isinstance(value, CallbackList):
        new_value = list(value)
    else:
        new_value = value

    # TODO: recursive (handle nested dicts and lists)

    return new_value


class Meta:
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

            self.from_dict(required_meta)
            self._write()

    def to_dict(self):
        d = {}
        for key in self._required_keys:
            value = getattr(self, key)
            d[key] = _convert_from_callback_classes_recursively(value)


        return d


    def from_dict(self, meta: dict):
        for key, value in meta.items():
            new_value = _convert_to_callback_classes_recursively(value, self._write)
            setattr(self, key, new_value)

        self._required_keys = meta.keys()
        return self


    def _write(self):
        self.storage[self.key] = bytes(json.dumps(self.to_dict()), "utf8")


    def _read(self):
        meta = json.loads(self.storage[self.key])
        return self.from_dict(meta)