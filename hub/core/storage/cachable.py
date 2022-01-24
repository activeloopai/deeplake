from abc import ABC
import json
from typing import Any, Dict


class Cachable(ABC):
    def __init__(self, buffer: bytes = None):
        if buffer:
            self.frombuffer(buffer)
        self.is_dirty = True

    @property
    def nbytes(self):
        # do not implement, each class should do this because it could be very slow if `tobytes` is called
        raise NotImplementedError

    def __getstate__(self) -> Dict[str, Any]:
        return self.__dict__

    def __setstate__(self, state: Dict[str, Any]):
        self.__dict__.update(state)

    def tobytes(self) -> bytes:
        d = {str(k): v for k, v in self.__getstate__().items()}
        return bytes(json.dumps(d, sort_keys=True, indent=4), "utf-8")

    def copy(self):
        return self.frombuffer(self.tobytes())

    @classmethod
    def frombuffer(cls, buffer: bytes):
        instance = cls()
        if len(buffer) > 0:
            instance.__setstate__(json.loads(buffer))
            instance.is_dirty = False
            return instance
        raise BufferError("Unable to instantiate the object as the buffer was empty.")
