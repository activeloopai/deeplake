from hub.core.storage.cachable import Cachable
import pickle


class VersionChunkList(Cachable):
    def __init__(self) -> None:
        self.chunks_in_commit = []

    @property
    def nbytes(self):
        return 4 * len(self.chunks_in_commit)  # TODO: set correct size

    def tobytes(self) -> bytes:
        return pickle.dumps(self.chunks_in_commit)

    @classmethod
    def frombuffer(cls, buffer: bytes):
        instance = cls()
        instance.chunks_in_commit = pickle.loads(buffer)
        return instance

    def append(self, item):
        self.chunks_in_commit.append(item)
