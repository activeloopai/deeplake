from typing import Any, Dict
from hub.core.meta.meta import Meta
from hub.core.index import Index
import posixpath


class DatasetMeta(Meta):
    def __init__(self):
        super().__init__()
        self.tensors = []
        self.groups = []
        self.tensor_names = {}
        self.hidden_tensors = []
        self.default_index = Index().to_json()

    @property
    def visible_tensors(self):
        return list(
            filter(
                lambda t: self.tensor_names[t] not in self.hidden_tensors,
                self.tensor_names.keys(),
            )
        )

    @property
    def nbytes(self):
        # TODO: can optimize this
        return len(self.tobytes())

    def __getstate__(self) -> Dict[str, Any]:
        d = super().__getstate__()
        d["tensors"] = self.tensors
        d["groups"] = self.groups
        d["tensor_names"] = self.tensor_names
        d["hidden_tensors"] = self.hidden_tensors
        d["default_index"] = self.default_index
        return d

    def add_tensor(self, name, key, hidden=False):
        if key not in self.tensors:
            self.tensor_names[name] = key
            self.tensors.append(key)
            if hidden:
                self.hidden_tensors.append(key)
            self.is_dirty = True

    def _hide_tensor(self, name):
        assert name in self.tensor_names
        if name not in self.hidden_tensors:
            self.hidden_tensors.append(self.tensor_names[name])
            self.is_dirty = True

    def add_group(self, name):
        if name not in self.groups:
            self.groups.append(name)
            self.is_dirty = True

    def delete_tensor(self, name):
        key = self.tensor_names.pop(name)
        self.tensors.remove(key)
        self.is_dirty = True

    def delete_group(self, name):
        self.groups = list(filter(lambda g: not g.startswith(name), self.groups))
        self.tensors = list(filter(lambda t: not t.startswith(name), self.tensors))
        self.hidden_tensors = list(
            filter(lambda t: not t.startswith(name), self.hidden_tensors)
        )
        tensor_names_keys = list(self.tensor_names.keys())
        for key in tensor_names_keys:
            if key.startswith(name):
                self.tensor_names.pop(key)
        self.is_dirty = True

    def rename_tensor(self, name, new_name):
        key = self.tensor_names.pop(name)
        self.tensor_names[new_name] = key
        self.is_dirty = True

    def rename_group(self, name, new_name):
        self.groups.remove(name)
        self.groups = list(
            map(
                lambda g: posixpath.join(new_name, posixpath.relpath(g, name))
                if (g == name or g.startswith(name + "/"))
                else g,
                self.groups,
            )
        )
        self.groups.append(new_name)
        self.is_dirty = True
