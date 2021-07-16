from hub.core.storage.lru_cache import LRUCache
from hub.util.json import validate_is_jsonable
from typing import Any, Dict
from hub.core.storage.cachable import CachableCallback, use_callback


_VALUE_ERROR_MSG = '`info.update` should be called with a single dictionary or **kwargs values. Example: `info.update({"key1": 1}, key2=2, key3=3)`'


class Info(CachableCallback):
    def __init__(self):
        """Contains **optional** key/values that datasets/tensors use for human-readability.
        See the `Meta` class for required key/values for datasets/tensors.

        Note:
            Since `Info` is rarely written to and mostly by the user, every modifier will call `cache[key] = self`.
            This is so the user doesn't have to call `flush`.
            Must call `initialize_callback_location` before using any methods.
        """

        self._info = {}
        super().__init__()

    @property
    def nbytes(self):
        # TODO: optimize this
        return len(self.tobytes())

    @use_callback(check_only=True)
    def __len__(self):
        return len(self._info)

    @use_callback(check_only=True)
    def __getstate__(self) -> Dict[str, Any]:
        return self._info

    def __setstate__(self, state: Dict[str, Any]):
        self._info = state

    @use_callback()
    def update(self, *args, **kwargs):
        """Updates info and synchronizes with cache. Inputs must be supported by JSON.
        A full list of supported value types can be found here: https://docs.python.org/3/library/json.html#json.JSONEncoder.

        Note:
            This method has the same functionality as `dict().update(...)` Reference: https://www.geeksforgeeks.org/python-dictionary-update-method/.
        """

        self._cache.check_readonly()
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


def load_info(info_key: str, cache: LRUCache):
    if info_key in cache:
        info = cache.get_cachable(info_key, Info)
    else:
        info = Info()
        info.initialize_callback_location(info_key, cache)

    return info
