from typing import Set
from hub.core.storage.cachable import Cachable


class CommitChunkSet(Cachable):
    """Stores set of chunks stored for a particular tensor in a commit."""

    def __init__(self) -> None:
        self.chunks: Set[str] = set()

    def tobytes(self) -> bytes:
        """Dumps self.chunks in csv format."""
        return bytes(",".join(self.chunks), "utf-8")

    @classmethod
    def frombuffer(cls, buffer: bytes):
        """Loads a CommitChunkSet from a buffer."""
        instance = cls()
        instance.chunks = set(buffer.decode("utf-8").split(","))
        return instance

    @property
    def nbytes(self) -> int:
        if not self.chunks:
            return 0
        return 8 + ((len(self.chunks) - 1) * 9)

    def add(self, chunk_name: str) -> None:
        """Adds a new chunk name to the CommitChunkSet."""
        self.chunks.add(chunk_name)
