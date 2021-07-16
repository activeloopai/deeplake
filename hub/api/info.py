from hub.util.json import validate_is_jsonable
from typing import Any
from hub.core.storage.cachable import CachableCallback, use_callback


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
    @use_callback(check_only=True)
    def nbytes(self):
        # TODO: optimize this
        return len(self.tobytes())

    @use_callback(check_only=True)
    def __len__(self):
        return len(self._info)

    @use_callback(check_only=True)
    def as_dict(self) -> dict:
        # TODO: docstring (INTERNAL USE ONLY!)

        # TODO: optimize this
        return {"_info": self._info.copy()}

    @use_callback()
    def update(self, *args, **kwargs):
        # TODO: docstring (mention jsonable)

        if len(args) > 1:
            raise ValueError(_VALUE_ERROR_MSG)  # TODO: exceptions.py

        if len(args) == 1:
            if not isinstance(args[0], dict):
                raise ValueError(_VALUE_ERROR_MSG)

            for k, v in args[0].items():
                validate_is_jsonable(k, v)

        for k, v in kwargs.items():
            validate_is_jsonable(k, v)

        self._info.update(*args, **kwargs)

    def __getattribute__(self, name: str) -> Any:
        """Allows access to info values using the `.` syntax. Example: `info.description`."""

        if name == "_info":
            return super().__getattribute__(name)
        if name in self._info:
            return self.__getitem__(name)
        return super().__getattribute__(name)

    def __getitem__(self, key: str):
        return self._info[key]

    def __str__(self):
        return self._info.__str__()

    def __repr__(self):
        return self._info.__repr__()
