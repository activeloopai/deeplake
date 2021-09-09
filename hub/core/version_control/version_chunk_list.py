from hub.core.storage.cachable import Cachable


class VersionChunkList(Cachable):
    def __init__(self) -> None:
        self.chunks = []

    @property
    def nbytes(self):
        if not self.chunks:
            return 14
        return 12 + (44 * len(self.chunks))

    def append(self, item):
        self.chunks.append(item)
