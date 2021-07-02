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
        s = json.dumps({"_version": self._version, **kwargs})
        self._storage[self._key] = s.encode(META_ENCODING)

    def load(self):
        self._check_readonly()
        buffer = self._storage[self._key]
        self.__dict__.update(json.loads(buffer.decode(META_ENCODING)))

    def as_readonly(self):
        if self._readonly:
            raise ReadOnlyModeError("This reference is already in readonly mode.")

        obj = copy(self)

        del obj._storage
        del obj._key
        obj._readonly = True

        return obj
