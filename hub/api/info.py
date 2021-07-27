from hub.core.storage.lru_cache import LRUCache
from typing import Any, Dict, Optional, Union, Sequence
from hub.core.storage.cachable import CachableCallback, use_callback


class Info(CachableCallback):
    def __init__(self):
        """Contains **optional** key/values that datasets/tensors use for human-readability.
        See the `Meta` class for required key/values for datasets/tensors.

        Note:
            Since `Info` is rarely written to and mostly by the user, every modifier will call `cache[key] = self`.
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
        """Store optional dataset/tensor information. Will be accessible after loading your data from a new script!
        Inputs must be supported by JSON.


        Note:
            This method has the same functionality as `dict().update(...)` Reference: https://www.geeksforgeeks.org/python-dictionary-update-method/.
            A full list of supported value types can be found here: https://docs.python.org/3/library/json.html#json.JSONEncoder.

        Examples:
            Normal update usage:
                >>> ds.info
                {}
                >>> ds.info.update(key=0)
                >>> ds.info
                {"key": 0}
                >>> ds.info.update({"key1": 5, "key2": [1, 2, "test"]})
                >>> ds.info
                {"key": 0, "key1": 5, "key2": [1, 2, "test"]}

            Alternate update usage:
                >>> ds.info
                {}
                >>> ds.info.update(list=[1, 2, "apple"])
                >>> ds.info
                {"list": [1, 2, "apple"]}
                >>> l = ds.info.list
                >>> l
                [1, 2, "apple"]
                >>> l.append(5)
                >>> l
                [1, 2, "apple", 5]
                >>> ds.info.update()  # required to be persistent!

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

    @use_callback()
    def delete(self, key: Optional[Union[Sequence[str], str]] = None):
        """Deletes a key or list of keys. If no key(s) is passed, all keys are deleted."""
        self._cache.check_readonly()
        if key is None:
            self._info.clear()
        elif isinstance(key, str):
            del self._info[key]
        elif isinstance(key, Sequence):
            for k in key:
                del self._info[k]
        else:
            raise KeyError(key)

    @use_callback()
    def __setitem__(self, key: str, value):
        self._cache.check_readonly()
        self._info[key] = value

    def __setattr__(self, key: str, value):
        if key in ("_key", "_cache", "_info"):
            object.__setattr__(self, key, value)
        else:
            self[key] = value

    def __getattr__(self, key: str):
        try:
            return object.__getattribute__(self, key)
        except AttributeError:
            return self[key]


def load_info(info_key: str, cache: LRUCache):
    if info_key in cache:
        info = cache.get_cachable(info_key, Info)
    else:
        info = Info()
        info.initialize_callback_location(info_key, cache)

    return info
