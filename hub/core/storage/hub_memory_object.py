from abc import ABC, abstractmethod
import json
from typing import Any, Dict


class HubMemoryObject(ABC):
    def __init__(self):
        self.is_dirty = True

    @property
    @abstractmethod
    def nbytes(self):
        """Returns the number of bytes in the object."""

    def __getstate__(self) -> Dict[str, Any]:
        return self.__dict__

    def __setstate__(self, state: Dict[str, Any]):
        self.__dict__.update(state)

    def tobytes(self) -> bytes:
        d = {str(k): v for k, v in self.__getstate__().items()}
        return bytes(json.dumps(d, sort_keys=True, indent=4), "utf-8")

    @classmethod
    def frombuffer(cls, buffer: bytes):
        instance = cls()
        if len(buffer) > 0:
            instance.__setstate__(json.loads(buffer))
            instance.is_dirty = False
            return instance
        raise BufferError("Unable to instantiate the object as the buffer was empty.")
