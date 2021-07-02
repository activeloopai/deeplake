from copy import deepcopy
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

    @property
    def version(self):
        return self._version

    def write(self, **kwargs):
        if self._readonly:
            raise ReadOnlyModeError()

        s = json.dumps({"version": self._version, **kwargs})
        self._storage[self._key] = s.encode(META_ENCODING)

    def load(self):
        if self._readonly:
            raise ReadOnlyModeError(
                "Loading is forbidden in readonly-mode because loading can modify this reference."
            )

        buffer = self._storage[self._key]
        self.__dict__.update(json.loads(buffer.decode(META_ENCODING)))

    def as_readonly(self):
        copy = deepcopy(self)

        del copy._storage
        del copy._key
        copy._readonly = True

        return copy
