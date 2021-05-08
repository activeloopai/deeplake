class ChunkSizeTooSmallError(Exception):
    def __init__(
        self,
        message="If the size of the last chunk is given, it must be smaller than the requested chunk size.",
    ):
        super().__init__(message)


class InvalidBytesRequestedError(Exception):
    def __init__(self):
        super().__init__(
            "The byte range provided is invalid. Ensure that start_byte <= end_byte and start_byte > 0 and end_byte > 0"
        )


# TODO Better S3 Exception handling
class S3GetError(Exception):
    """Catchall for all errors encountered while working getting an object from S3"""

    pass


class S3SetError(Exception):
    """Catchall for all errors encountered while working setting an object in S3"""

    pass


class S3DeletionError(Exception):
    """Catchall for all errors encountered while working deleting an object in S3"""

    pass


class S3ListError(Exception):
    """Catchall for all errors encountered while retrieving a list of objects present in S3"""

    pass
