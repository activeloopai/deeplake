from typing import Any, Sequence


class ChunkSizeTooSmallError(Exception):
    def __init__(
        self,
        message="If the size of the last chunk is given, it must be smaller than the requested chunk size.",
    ):
        super().__init__(message)


class TensorMetaMismatchError(Exception):
    def __init__(self, meta_key: str, expected: Any, actual: Any):
        super().__init__(
            "Meta value for {} expected {} but got {}.".format(
                meta_key, str(expected), str(actual)
            )
        )


class TensorInvalidSampleShapeError(Exception):
    def __init__(self, message: str, shape: Sequence[int]):
        super().__init__("{} Incoming sample shape: {}".format(message, str(shape)))


class TensorMetaMissingKey(Exception):
    def __init__(self, key: str, meta: dict):
        super().__init__("Key {} missing from tensor meta {}.".format(key, str(meta)))


class TensorMetaInvalidValue(Exception):
    def __init__(self, key: str, value: Any, explanation: str = ""):
        super().__init__(
            "Invalid value {} for tensor meta key {}. {}".format(
                str(value), key, explanation
            )
        )


class TensorDoesNotExistError(KeyError):
    def __init__(self, tensor_name: str):
        super().__init__("Tensor {} does not exist.".format(tensor_name))


class TensorAlreadyExistsError(Exception):
    def __init__(self, key: str):
        super().__init__("Tensor {} already exists.".format(key))


class DynamicTensorNumpyError(Exception):
    def __init__(self, key: str, index):
        super().__init__(
            "Tensor {} with index = {} is dynamically shaped and cannot be converted into a `np.ndarray`. \
            Try setting the parameter `aslist=True`".format(
                key, str(index)
            )
        )


class InvalidShapeIntervalError(Exception):
    def __init__(
        self, message: str, lower: Sequence[int] = None, upper: Sequence[int] = None
    ):
        s = message

        if lower is not None:
            s += " lower={}".format(str(lower))

        if upper is not None:
            s += " upper={}".format(str(upper))

        super().__init__(s)


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


class ModuleNotInstalledException(Exception):
    def __init__(self, message):
        super().__init__(message)


class LoginException(Exception):
    def __init__(
        self,
        message="Error while logging in, invalid auth token. Please try logging in again.",
    ):
        super().__init__(message)


class ImproperDatasetInitialization(Exception):
    def __init__(self):
        super().__init__(
            "Exactly one argument out of 'path' and 'storage' should be provided."
        )


class InvalidHubPathException(Exception):
    def __init__(self, path):
        super().__init__(
            f"The Dataset's path is an invalid Hub path. It should be of the form hub://username/dataset got {path}."
        )


class PathNotEmptyException(Exception):
    def __init__(self):
        super().__init__(
            f"The url specified doesn't point to a Hub Dataset and the folder isn't empty. Please use a url that points to an existing Hub Dataset or an empty folder."
        )


# Exceptions encountered while interection with the Hub backend
class AuthenticationException(Exception):
    def __init__(self, message="Authentication failed. Please try logging in again."):
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
        message = (
            f"Invalid Request. One or more request parameters is incorrect.\n{message}"
        )
        super().__init__(message)


class OverLimitException(Exception):
    def __init__(
        self,
        message="You are over the allowed limits for this operation.",
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


class InvalidTokenException(Exception):
    def __init__(self, message="The authentication token is empty."):
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


class InvalidCompressor(Exception):
    def __init__(self, available_compressors):
        super().__init__(
            f"Compressor is not supported. Supported compressions: {available_compressors}"
        )


class InvalidImageDimensions(Exception):
    def __init__(self, actual_dims, expected_dims):
        super().__init__(
            f"The shape length {actual_dims} of the given array should "
            f"be greater than the number of expected dimensions {expected_dims}"
        )


class ReadOnlyModeError(Exception):
    def __init__(self):
        super().__init__("Modification when in read-only mode is not supported!")
