from hub.util.immutability import (
    recursively_parse_as_immutable,
    validate_can_be_parsed_as_immutable,
)
from typing import Any
from hub.util.storage_callback import CachableCallback, callback


_VALUE_ERROR_MSG = '`info.update` should be called with a single dictionary or **kwargs values. Example: `info.update({"key1": 1}, key2=2, key3=3)`'


class Info(CachableCallback):
    def __init__(self):
        """Contains **optional** key/values that datasets/tensors use for human-readability.
        See the `Meta` class for required key/values for datasets/tensors.

        Note:
            Since `Info` is rarely written to and mostly by the user, every modifier will call `storage[key] = self`.
            This is so the user doesn't have to call `flush`.
            Must call `initialize_callback_location` before using any methods.
        """

        self._info = {}
        super().__init__()

    @property
    @callback(check_only=True)
    def nbytes(self):
        # TODO: optimize this
        return len(self.tobytes())

    @callback(check_only=True)
    def __len__(self):
        return len(self._info)

    @callback(check_only=True)
    def as_dict(self) -> dict:
        # TODO: docstring (INTERNAL USE ONLY!)

        # TODO: optimize this

        return {"_info": self._info.copy()}

    @callback()
    def update(self, *args, **kwargs):
        # TODO: convert everything to immutable structures

        if len(args) > 1:
            raise ValueError(_VALUE_ERROR_MSG)  # TODO: exceptions.py

        if len(args) == 1:
            if not isinstance(args[0], dict):
                raise ValueError(_VALUE_ERROR_MSG)

            for v in args[0].values():
                validate_can_be_parsed_as_immutable(v, recursive=True)

        for v in kwargs.values():
            validate_can_be_parsed_as_immutable(v, recursive=True)

        self._info.update(*args, **kwargs)

    def __getattribute__(self, name: str) -> Any:
        """Allows access to info values using the `.` syntax. Example: `info.description`."""

        if name == "_info":
            return super().__getattribute__(name)
        if name in self._info:
            return self.__getitem__(name)
        return super().__getattribute__(name)

    def __getitem__(self, key: str):
        # TODO: docstring (immutability)

        value = self._info[key]

        # TODO: return immutable (tuples and stuff)
        return recursively_parse_as_immutable(value)
