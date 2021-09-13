from typing import List
from hub.core.storage.cachable import Cachable


class CommitChunkList(Cachable):
    """Stores list of chunks stored for a particular tensor in a commit."""
    def __init__(self) -> None:
        self.chunks: List[str] = []

    def tobytes(self) -> bytes:
        """Dumps self.chunks in csv format."""
        return bytes(",".join(self.chunks), "utf-8")

    @classmethod
    def frombuffer(cls, buffer: bytes):
        """Loads a VersionChunkList from a buffer."""
        instance = cls()
        instance.chunks = buffer.decode("utf-8").split(",")
        return instance

    @property
    def nbytes(self) -> int:
        if not self.chunks:
            return 0
        return 8 + ((len(self.chunks) - 1) * 9)

    def append(self, chunk_name: str) -> None:
        """Adds a new chunk name to the VersionChunkList."""
        self.chunks.append(chunk_name)
