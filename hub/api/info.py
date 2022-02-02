from sqlite3 import NotSupportedError
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
            if not self._dataset.version_state["commit_node"].is_head_node:
                raise NotSupportedError("Cannot modify info from a non-head commit.")
            self.is_dirty = True

    def end_write(self):
        if self._dataset is not None:
            storage = self._dataset.storage
            storage.maybe_flush()

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
        self.end_write()

    def __delitem__(self, key):
        self.prepare_for_write()
        del self._info[key]
        self.end_write()

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
            self.end_write()

    def get(self, key, default=None):
        return self._info.get(key, default)

    def setdefault(self, key, default=None):
        self.prepare_for_write()
        ret = self._info.setdefault(key, default)
        self.end_write()
        return ret

    def clear(self):
        self.prepare_for_write()
        self._info.clear()
        self.end_write()

    def pop(self, key, default=None):
        self.prepare_for_write()
        popped = self._info.pop(key, default)
        self.end_write()
        return popped

    def popitem(self):
        self.prepare_for_write()
        popped = self._info.popitem()
        self.end_write()
        return popped

    def update(self, *args, **kwargs):
        self.prepare_for_write()
        self._info.update(*args, **kwargs)
        self.end_write()

    def keys(self):
        return self._info.keys()

    def values(self):
        return self._info.values()

    def items(self):
        return self._info.items()

    def replace_with(self, d):
        self.prepare_for_write()
        self._info.clear()
        self._info.update(d)
        self.end_write()

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
