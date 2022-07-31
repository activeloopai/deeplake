from hub.core.storage.lru_cache import LRUCache
from hub.util.exceptions import InfoError
from hub.core.storage.hub_memory_object import HubMemoryObject
from typing import Any, Dict


class Info(HubMemoryObject):
    def __init__(self):
        self._info = {}
        self._dataset = None

        # the key to info in case of Tensor Info, None in case of Dataset Info
        self._key = None
        self.is_dirty = False

    def __enter__(self):
        from hub.core.tensor import Tensor

        ds = self._dataset
        key = self._key
        if ds is not None:
            ds.storage.check_readonly()
            if not ds.version_state["commit_node"].is_head_node:
                raise InfoError("Cannot modify info from a non-head commit.")
            if key:
                Tensor(key, ds).chunk_engine.commit_diff.modify_info()
            else:
                ds._dataset_diff.modify_info()
            self.is_dirty = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._dataset is not None:
            self._dataset.storage.maybe_flush()

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
        """Allows access to info values using the `.` syntax. Example: ``info.description``."""

        if name == "_info":
            return super().__getattribute__(name)
        if name in self._info:
            return self.__getitem__(name)
        return super().__getattribute__(name)

    # implement all the methods of dictionary
    def __getitem__(self, key: str):
        return self._info[key]

    def __str__(self):
        return self._info.__str__()

    def __repr__(self):
        return self._info.__repr__()

    def __setitem__(self, key, value):
        with self:
            self._info[key] = value

    def __delitem__(self, key):
        with self:
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
        if key in {"_info", "_dataset", "_key", "is_dirty"}:
            object.__setattr__(self, key, value)
        else:
            with self:
                self[key] = value

    def get(self, key, default=None):
        return self._info.get(key, default)

    def setdefault(self, key, default=None):
        with self:
            ret = self._info.setdefault(key, default)
        return ret

    def clear(self):
        with self:
            self._info.clear()

    def pop(self, key, default=None):
        with self:
            popped = self._info.pop(key, default)
        return popped

    def popitem(self):
        with self:
            popped = self._info.popitem()
        return popped

    def update(self, *args, **kwargs):
        with self:
            self._info.update(*args, **kwargs)

    def keys(self):
        return self._info.keys()

    def values(self):
        return self._info.values()

    def items(self):
        return self._info.items()

    def replace_with(self, d):
        with self:
            self._info.clear()
            self._info.update(d)

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


def load_info(path, dataset, key=None) -> Info:
    storage: LRUCache = dataset.storage

    try:
        info = storage.get_hub_object(path, Info)
    except KeyError:
        info = Info()

    info._dataset = dataset
    info._key = key
    storage.register_hub_object(path, info)
    return info
