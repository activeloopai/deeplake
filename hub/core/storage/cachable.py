from abc import ABC
import json


class Cachable(ABC):
    """Cachable objects can be stored in memory cache without being converted into bytes until flushed.

    For example, `Chunk` is a subclass of `Cachable`. This enables us to update chunk state without serializing
    and deserializing until it's finalized and ready to be flushed from the cache.
    """

    def __init__(self, buffer: bytes = None):
        if buffer:
            self.frombuffer(buffer)

    def __len__(self):
        return len(self.tobytes())

    def tobytes(self) -> bytes:
        return bytes(json.dumps(self.__dict__), "utf-8")

    @classmethod
    def frombuffer(cls, buffer: bytes):
        instance = cls()
        instance.__dict__.update(json.loads(buffer))
        return instance
