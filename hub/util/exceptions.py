from typing import Any


class ChunkSizeTooSmallError(Exception):
    def __init__(
        self,
        message="If the size of the last chunk is given, it must be smaller than the requested chunk size.",
    ):
        super().__init__(message)


class TensorNotFoundError(KeyError):
    def __init__(self, tensor_name: str, dataset_path: str):
        super().__init__(
            "Tensor {} not found in dataset {}".format(tensor_name, dataset_path)
        )


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


class LoginException:
    def __init__(
        self,
        message="Error while logging in, invalid auth token. Try logging in again.",
    ):
        super().__init__(message)


# Exceptions encountered while interection with the Hub backend
class AuthenticationException(Exception):
    def __init__(self, message="Authentication failed. Please login again."):
        super().__init__(message)


class AuthorizationException(Exception):
    def __init__(
        self,
        message="You are not authorized to access this resource on Activeloop Server.",
    ):
        super().__init__(message)


class ResourceNotFoundException(Exception):
    def __init__(
        self,
        message="The resource you are looking for was not found. Check if the name or id is correct.",
    ):
        super().__init__(message)


class BadRequestException(Exception):
    def __init__(self, message):
        message = f"One or more request parameters is incorrect\n{message}"
        super().__init__(message)


class OverLimitException(Exception):
    def __init__(
        self,
        message="You are over the allowed limits for this operation. Consider upgrading your account.",
    ):
        super().__init__(message)


class ServerException(Exception):
    def __init__(self, message="Internal Activeloop server error."):
        super().__init__(message)


class BadGatewayException(Exception):
    def __init__(self, message="Invalid response from Activeloop server."):
        super().__init__(message)


class GatewayTimeoutException(Exception):
    def __init__(self, message="Activeloop server took too long to respond."):
        super().__init__(message)


class WaitTimeoutException(Exception):
    def __init__(self, message="Timeout waiting for server state update."):
        super().__init__(message)


class LockedException(Exception):
    def __init__(self, message="The resource is currently locked."):
        super().__init__(message)


class UnexpectedStatusCodeException(Exception):
    def __init__(self, message):
        super().__init__(message)


# TODO Better S3 Exception handling
class S3GetError(Exception):
    """Catchall for all errors encountered while working getting an object from S3"""


class S3SetError(Exception):
    """Catchall for all errors encountered while working setting an object in S3"""


class S3DeletionError(Exception):
    """Catchall for all errors encountered while working deleting an object in S3"""


class S3ListError(Exception):
    """Catchall for all errors encountered while retrieving a list of objects present in S3"""
