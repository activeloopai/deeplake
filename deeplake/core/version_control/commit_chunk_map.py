from typing import Dict
from deeplake.core.storage.deeplake_memory_object import DeepLakeMemoryObject


class CommitChunkMap(DeepLakeMemoryObject):
    def __init__(self) -> None:
        self.is_dirty = False
        self.chunks: Dict[str, str] = dict()

    def tobytes(self) -> bytes:
        return bytes("\n".join([f"{k},{v}" for k, v in self.chunks.items()]), "utf-8")

    @classmethod
    def frombuffer(cls, buffer: bytes):
        instance = cls()
        if buffer:
            instance.chunks = dict([line.split(",") for line in buffer.decode("utf-8").split("\n")])
        instance.is_dirty = False
        return instance

    @property
    def nbytes(self) -> int:
        if not self.chunks:
            return 0
        return len(self.chunks) * (40 + 16 + 1 + 1) - 1

    def add(self, chunk_name: str, commit_id: str) -> None:
        self.chunks[chunk_name] = commit_id
        self.is_dirty = True
