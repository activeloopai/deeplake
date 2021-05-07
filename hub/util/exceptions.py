class ChunkSizeTooSmallError(Exception):
    def __init__(
        self,
        message="If the size of the last chunk is given, it must be smaller than the requested chunk size.",
    ):
        super().__init__(message)
