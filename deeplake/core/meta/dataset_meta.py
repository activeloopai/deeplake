from typing import Any, Dict
from deeplake.core.meta.meta import Meta
from deeplake.core.index import Index
import posixpath

from deeplake.util.path import relpath


class DatasetMeta(Meta):
    """Stores dataset metadata."""

    def __init__(self):
        super().__init__()
        self.tensors = []
        self.groups = []
        self.tensor_names = {}
        self.hidden_tensors = []
        self.default_index = Index().to_json()

    @property
    def visible_tensors(self):
        """Returns list of tensors that are not hidden."""
        return list(
            filter(
                lambda t: self.tensor_names[t] not in self.hidden_tensors,
                self.tensor_names.keys(),
            )
        )

    @property
    def nbytes(self):
        """Returns size of the metadata stored in bytes."""
        # TODO: can optimize this
        return len(self.tobytes())

    def __getstate__(self) -> Dict[str, Any]:
        d = super().__getstate__()
        d["tensors"] = self.tensors.copy()
        d["groups"] = self.groups.copy()
        d["tensor_names"] = self.tensor_names.copy()
        d["hidden_tensors"] = self.hidden_tensors.copy()
        d["default_index"] = self.default_index.copy()
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    def add_tensor(self, name, key, hidden=False):
        """Reflect addition of tensor in dataset's meta."""
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
        """Reflect addition of tensor group in dataset's meta."""
        if name not in self.groups:
            self.groups.append(name)
            self.is_dirty = True

    def delete_tensor(self, name):
        """Reflect tensor deletion in dataset's meta."""
        key = self.tensor_names.pop(name)
        self.tensors.remove(key)
        try:
            self.hidden_tensors.remove(key)
        except ValueError:
            pass
        self.is_dirty = True

    def delete_group(self, name):
        """Reflect removal of a tensor group in dataset's meta."""
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
        """Reflect a tensor rename in dataset's meta."""
        key = self.tensor_names.pop(name)
        self.tensor_names[new_name] = key
        self.is_dirty = True

    def rename_group(self, name, new_name):
        """Reflect renaming a tensor group in dataset's meta."""
        self.groups.remove(name)
        self.groups = list(
            map(
                lambda g: posixpath.join(new_name, relpath(g, name))
                if (g == name or g.startswith(name + "/"))
                else g,
                self.groups,
            )
        )
        self.groups.append(new_name)
        self.is_dirty = True
