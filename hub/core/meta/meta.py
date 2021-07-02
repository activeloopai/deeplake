from copy import copy
from hub.util.exceptions import ReadOnlyModeError
from hub.constants import META_ENCODING
from hub.core.storage.provider import StorageProvider
import hub
import json


class Meta:
    def __init__(self, key: str, storage: StorageProvider):
        self._key = key
        self._storage = storage

        self._version = hub.__version__
        self._readonly = False

    def _check_readonly(self):
        if self._readonly:
            raise ReadOnlyModeError(
                "Forbidden operation in readonly-mode because it may modify this reference."
            )

    @property
    def version(self):
        return self._version

    def write(self, **kwargs):
        self._check_readonly()
        s = json.dumps({"version": self._version, **kwargs})
        self._storage[self._key] = s.encode(META_ENCODING)

    def load(self):
        self._check_readonly()
        buffer = self._storage[self._key]
        meta_dict = json.loads(buffer.decode(META_ENCODING))

        # add `_` to all variable names (setting private fields instead of their @properties)
        for k, v in meta_dict.copy().items():
            del meta_dict[k]
            meta_dict[f"_{k}"] = v

        self.__dict__.update(meta_dict)

    def as_readonly(self):
        if self._readonly:
            raise ReadOnlyModeError("This reference is already in readonly mode.")

        obj = copy(self)

        del obj._storage
        del obj._key
        obj._readonly = True

        return obj
