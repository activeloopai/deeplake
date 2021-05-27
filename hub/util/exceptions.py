from typing import Any
from hub.core.compression import AVAILABLE_COMPRESSORS


class ChunkSizeTooSmallError(Exception):
    def __init__(
        self,
        message="If the size of the last chunk is given, it must be smaller than the requested chunk size.",
    ):
        super().__init__(message)


class TensorNotFoundError(KeyError):
    def __init__(self, tensor_name: str):
        super().__init__("Tensor {} not found in dataset.".format(tensor_name))


class InvalidKeyTypeError(TypeError):
    def __init__(self, item: Any):
        super().__init__(
            "Item {} is of type {} is not a valid key".format(
                str(item), type(item).__name__
            )
        )


class UnsupportedTensorTypeError(TypeError):
    def __init__(self, item: Any):
        super().__init__(
            "Key of type {} is not currently supported to convert to a tensor.".format(
                type(item).__name__
            )
        )


class InvalidBytesRequestedError(Exception):
    def __init__(self):
        super().__init__(
            "The byte range provided is invalid. Ensure that start_byte <= end_byte and start_byte > 0 and end_byte > 0"
        )


class ProviderListEmptyError(Exception):
    def __init__(self):
        super().__init__(
            "The provider_list passed to get_cache_chain needs to have 1 or more elements."
        )


class DirectoryAtPathException(Exception):
    def __init__(self):
        super().__init__(
            "The provided path is incorrect for this operation, there is a directory at the path. Provide a path to a file."
        )


class FileAtPathException(Exception):
    def __init__(self, path):
        super().__init__(
            f"Expected a directory at path {path} but found a file instead."
        )


class ProviderSizeListMismatch(Exception):
    def __init__(self):
        super().__init__("Ensure that len(size_list) + 1 == len(provider_list)")


# TODO Better S3 Exception handling
class S3GetError(Exception):
    """Catchall for all errors encountered while working getting an object from S3"""


class S3SetError(Exception):
    """Catchall for all errors encountered while working setting an object in S3"""


class S3DeletionError(Exception):
    """Catchall for all errors encountered while working deleting an object in S3"""


class S3ListError(Exception):
    """Catchall for all errors encountered while retrieving a list of objects present in S3"""


class InvalidCompressor(Exception):
    def __init__(self):
        super().__init__(
            f"Compressor is not supported. Supported compressions: {AVAILABLE_COMPRESSORS}"
        )
