from hub.util.version_control import auto_checkout
from hub.core.storage.hub_memory_object import HubMemoryObject
from typing import Any, Dict, Optional


class Info(HubMemoryObject):
    def __init__(self):
        self._info = {}
        self._dataset = None
        super().__init__()

    def prepare_for_write(self):
        if self._dataset is not None:
            storage = self._dataset.storage
            storage.check_readonly()
            auto_checkout(self._dataset)
            self.is_dirty = True

    @property
    def nbytes(self):
        return len(self.tobytes())

    def __len__(self):
        return len(self._info)

    def __getstate__(self) -> Dict[str, Any]:
        return self._info

    def __setstate__(self, state: Dict[str, Any]):
        self._info = state

    def __getattribute__(self, name: str) -> Any:
        """Allows access to info values using the `.` syntax. Example: `info.description`."""

        if name == "_info":
            return super().__getattribute__(name)
        if name in self._info:
            return self.__getitem__(name)
        return super().__getattribute__(name)

    # implement all the methods of dictionary
    def __getitem__(self, key: str):
        return self._info[key]

    def get(self, key: str, default: Optional[Any] = None):
        return self._info.get(key, default)

    def __str__(self):
        return self._info.__str__()

    def __repr__(self):
        return self._info.__repr__()

    def __setitem__(self, key, value):
        self.prepare_for_write()
        self._info[key] = value

    def __delitem__(self, key):
        self.prepare_for_write()
        del self._info[key]

    def __contains__(self, key):
        return key in self._info

    def __iter__(self):
        return iter(self._info)

    def __getattr__(self, key):
        try:
            return object.__getattribute__(self, key)
        except AttributeError:
            if key == "_info":
                self._info = {}
                return self._info
            return self[key]

    def __setattr__(self, key: str, value):
        if key in {"_info", "_dataset", "is_dirty"}:
            object.__setattr__(self, key, value)
        else:
            self.prepare_for_write()
            self[key] = value

    def get(self, key, default=None):
        return self._info.get(key, default)

    def setdefault(self, key, default=None):
        self.prepare_for_write()
        return self._info.setdefault(key, default)

    def clear(self):
        self.prepare_for_write()
        self._info.clear()

    def pop(self, key, default=None):
        self.prepare_for_write()
        return self._info.pop(key, default)

    def popitem(self):
        self.prepare_for_write()
        return self._info.popitem()

    def update(self, *args, **kwargs):
        self.prepare_for_write()
        self._info.update(*args, **kwargs)

    def keys(self):
        return self._info.keys()

    def values(self):
        return self._info.values()

    def items(self):
        return self._info.items()

    # the below methods are used by cloudpickle dumps
    def __origin__(self):
        return None

    def __values__(self):
        return None

    def __type__(self):
        return None

    def __union_params__(self):
        return None

    def __tuple_params__(self):
        return None

    def __result__(self):
        return None

    def __args__(self):
        return None


def load_info(key, dataset):
    storage = dataset.storage
    info = storage.get_hub_object(key, Info) if key in storage else Info()
    info._dataset = dataset
    return info
