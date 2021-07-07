from hub.htypes import HTYPE_CONFIGURATIONS
from hub.constants import SUPPORTED_COMPRESSIONS
from typing import Any, List, Sequence, Tuple


class TensorInvalidSampleShapeError(Exception):
    def __init__(self, message: str, shape: Sequence[int]):
        super().__init__("{} Incoming sample shape: {}".format(message, str(shape)))


class TensorMetaMissingKey(Exception):
    def __init__(self, key: str, meta: dict):
        super().__init__("Key {} missing from tensor meta {}.".format(key, str(meta)))


class TensorDoesNotExistError(KeyError):
    def __init__(self, tensor_name: str):
        super().__init__("Tensor {} does not exist.".format(tensor_name))


class TensorAlreadyExistsError(Exception):
    def __init__(self, key: str):
        super().__init__("Tensor {} already exists.".format(key))


class DynamicTensorNumpyError(Exception):
    def __init__(self, key: str, index, property_key: str):
        super().__init__(
            f"Tensor {key} with index = {str(index)} is has a dynamic '{property_key}' and cannot be converted into a `np.ndarray`. Try setting the parameter `aslist=True`"
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


class CompressionError(Exception):
    pass


class UnsupportedCompressionError(CompressionError):
    def __init__(self, compression: str):
        super().__init__(
            f"Compression '{compression}' is not supported. Supported compressions: {SUPPORTED_COMPRESSIONS}."
        )


class SampleCompressionError(CompressionError):
    def __init__(
        self, sample_shape: Tuple[int, ...], compression_format: str, message: str
    ):
        super().__init__(
            f"Could not compress a sample with shape {str(sample_shape)} into '{compression_format}'. Raw error output: '{message}'.",
        )


class SampleDecompressionError(CompressionError):
    def __init__(self):
        super().__init__(
            f"Could not decompress sample buffer into an array. Either the sample's buffer is corrupted, or it is in an unsupported format. Supported compressions: {SUPPORTED_COMPRESSIONS}."
        )


class InvalidImageDimensions(Exception):
    def __init__(self, actual_dims, expected_dims):
        super().__init__(
            f"The shape length {actual_dims} of the given array should "
            f"be greater than the number of expected dimensions {expected_dims}"
        )


class TensorUnsupportedSampleType(Exception):
    def __init__(self) -> None:
        super().__init__(
            f"Unable to append sample. Please specify numpy array, sequence of numpy arrays"
            "or resulting dictionary from .read() to be added to the tensor"
        )


class MetaError(Exception):
    pass


class MetaDoesNotExistError(MetaError):
    def __init__(self, key: str):
        super().__init__(
            f"A meta (key={key}) cannot be instantiated without `required_meta` when it does not exist yet. \
            If you are trying to read the meta, heads up: it didn't get written."
        )


class MetaAlreadyExistsError(MetaError):
    def __init__(self, key: str, required_meta: dict):
        super().__init__(
            f"A meta (key={key}) cannot be instantiated with `required_meta` when it already exists. \
            If you are trying to write the meta, heads up: it already got written (required_meta={required_meta})."
        )


class MetaInvalidKey(MetaError):
    def __init__(self, name: str, available_keys: List[str]):
        super().__init__(
            f'"{name}" is an invalid key for meta (`meta_object.{name}`). \
            Maybe a typo? Available keys: {str(available_keys)}'
        )


class MetaInvalidRequiredMetaKey(MetaError):
    def __init__(self, key: str, subclass_name: str):
        super().__init__(
            f"'{key}' should not be passed in `required_meta` (it is probably automatically set). \
            This means the '{subclass_name}' class was constructed improperly."
        )


class TensorMetaInvalidHtype(MetaError):
    def __init__(self, htype: str, available_htypes: Sequence[str]):
        super().__init__(
            f"Htype '{htype}' does not exist. Available htypes: {str(available_htypes)}"
        )


class TensorMetaInvalidHtypeOverwriteValue(MetaError):
    def __init__(self, key: str, value: Any, explanation: str = ""):
        super().__init__(
            "Invalid value {} for tensor meta key {}. {}".format(
                str(value), key, explanation
            )
        )


class TensorMetaInvalidHtypeOverwriteKey(MetaError):
    def __init__(self, htype: str, key: str, available_keys: Sequence[str]):
        super().__init__(
            f"Htype '{htype}' doesn't have a key for '{key}'. Available keys: {str(available_keys)}"
        )


class TensorDtypeMismatchError(MetaError):
    def __init__(self, expected: str, actual: str, htype: str):
        msg = f"Dtype was expected to be '{expected}' instead it was '{actual}'. If you called `create_tensor` explicitly with `dtype`, your samples should also be of that dtype."

        # TODO: we may want to raise this error at the API level to determine if the user explicitly overwrote the `dtype` or not. (to make this error message more precise)
        # TODO: because if the user uses `dtype=np.uint8`, but the `htype` the tensor is created with has it's default dtype set as `uint8` also, then this message is ambiguous
        htype_dtype = HTYPE_CONFIGURATIONS[htype].get("dtype", None)
        if htype_dtype is not None and htype_dtype == expected:
            msg += f" Htype '{htype}' expects samples to have dtype='{htype_dtype}'."
            super().__init__("")

        super().__init__(msg)


class ReadOnlyModeError(Exception):
    def __init__(self, custom_message: str = None):
        if custom_message is None:
            custom_message = "Modification when in read-only mode is not supported!"
        super().__init__(custom_message)


class TransformError(Exception):
    pass


class InvalidTransformOutputError(TransformError):
    def __init__(self, item):
        super().__init__(
            f"The output of each step in a transformation should be either dictionary or a list/tuple of dictionaries, found {type(item)}."
        )


class InvalidInputDataError(TransformError):
    def __init__(self, message):
        super().__init__(
            f"The data_in to transform is invalid. It should support {message} operation."
        )


class UnsupportedSchedulerError(TransformError):
    def __init__(self, scheduler):
        super().__init__(
            f"Hub transform currently doesn't support {scheduler} scheduler."
        )


class TensorMismatchError(TransformError):
    def __init__(self, tensors, output_keys):
        super().__init__(
            f"One or more of the outputs generated during transform contain different tensors than the ones present in the output 'ds_out' provided to transform.\n "
            f"Tensors in ds_out: {tensors}\n Tensors in output sample: {output_keys}"
        )


class InvalidOutputDatasetError(TransformError):
    def __init__(self):
        super().__init__(
            "One or more tensors of the ds_out have different lengths. Transform only supports ds_out having same number of samples for each tensor (This includes empty datasets that have 0 samples per tensor)."
        )


class MemoryDatasetNotSupportedError(TransformError):
    def __init__(self, scheduler):
        super().__init__(
            f"Transforms with ds_out having base storage as MemoryProvider are only supported in threaded mode. Current mode is {scheduler}."
        )


class DatasetUnsupportedPytorch(Exception):
    def __init__(self, reason):
        super().__init__(
            f"The Dataset object passed to Pytorch is incompatible. Reason: {reason}"
        )


class CorruptedMetaError(Exception):
    pass


class ChunkEngineError(Exception):
    pass


class FullChunkError(ChunkEngineError):
    pass


class ChunkIdEncoderError(ChunkEngineError):
    pass


class ChunkSizeTooSmallError(ChunkEngineError):
    def __init__(
        self,
        message="If the size of the last chunk is given, it must be smaller than the requested chunk size.",
    ):
        super().__init__(message)
