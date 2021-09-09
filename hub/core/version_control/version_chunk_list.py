from typing import List
from hub.core.storage.cachable import Cachable


class VersionChunkList(Cachable):
    def __init__(self) -> None:
        self.chunks: List[str] = []

    def tobytes(self) -> bytes:
        """Dump self.chunks in csv format"""
        return bytes(",".join(self.chunks), "utf-8")

    @classmethod
    def frombuffer(cls, buffer: bytes):
        """Load a VersionChunkList from a buffer"""
        instance = cls()
        instance.chunks = buffer.decode("utf-8").split(",")
        return instance

    @property
    def nbytes(self):
        if not self.chunks:
            return 0
        return 8 + ((len(self.chunks) - 1) * 9)

    def append(self, item):
        self.chunks.append(item)
