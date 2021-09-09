from hub.core.storage.cachable import Cachable


class VersionChunkList(Cachable):
    def __init__(self) -> None:
        self.chunks_in_commit = []

    @property
    def nbytes(self):
        return 4 * len(self.chunks_in_commit)  # TODO: set correct size

    def append(self, item):
        self.chunks_in_commit.append(item)
