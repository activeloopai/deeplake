from typing import Set
from deeplake.core.storage.deeplake_memory_object import DeepLakeMemoryObject


class CommitChunkSet(DeepLakeMemoryObject):
    """Stores set of chunks stored for a particular tensor in a commit."""

    def __init__(self) -> None:
        self.is_dirty = False
        self.chunks: Set[str] = set()

    def tobytes(self) -> bytes:
        """Dumps self.chunks in csv format."""
        return bytes(",".join(self.chunks), "utf-8")

    @classmethod
    def frombuffer(cls, buffer: bytes):
        """Loads a CommitChunkSet from a buffer."""
        instance = cls()
        if buffer:
            instance.chunks = set(buffer.decode("utf-8").split(","))
        instance.is_dirty = False
        return instance

    @property
    def nbytes(self) -> int:
        if not self.chunks:
            return 0
        return 8 + ((len(self.chunks) - 1) * 9)

    def add(self, chunk_name: str) -> None:
        """Adds a new chunk name to the CommitChunkSet."""
        self.chunks.add(chunk_name)
        self.is_dirty = True
