import numpy as np
import deeplake
from typing import Any, List, Sequence, Tuple, Optional, Union


class ExternalCommandError(Exception):
    def __init__(self, command: str, status: int):
        super().__init__(
            f'Status for command "{command}" was "{status}", expected to be "0".'
        )


class KaggleError(Exception):
    message: str = ""


class KaggleMissingCredentialsError(KaggleError):
    def __init__(self, env_var_name: str):
        super().__init__(
            "Could not find %s in environment variables. Try setting them or providing the `credentials` argument. More information on how to get kaggle credentials: https://www.kaggle.com/docs/api"
            % env_var_name
        )


class KaggleDatasetAlreadyDownloadedError(KaggleError):
    def __init__(self, tag: str, path: str):
        self.message = f"Kaggle dataset {tag} already exists at {path}. You can get rid of this error by setting the `exist_ok` parameter to `True`."
        super().__init__(self.message)


class InvalidPathException(Exception):
    def __init__(self, directory):
        super().__init__(
            f"Dataset's path is an invalid path. It should be a valid local directory got {directory}."
        )


class AutoCompressionError(Exception):
    def __init__(self, directory):
        super().__init__(
            f"Auto compression could not run on {directory}. The directory doesn't contain any files."
        )


class InvalidFileExtension(Exception):
    def __init__(self, directory):
        super().__init__(
            f"Missing file with extension in {directory}. Expected a valid file extension got None."
        )


class SamePathException(Exception):
    def __init__(self, directory):
        super().__init__(
            f"Dataset source and destination path are same '{directory}'. Source and destination cannot be same for dataset ingestion, try setting different paths."
        )


class TensorInvalidSampleShapeError(Exception):
    def __init__(self, shape: Sequence[int], expected_dims: int):
        super().__init__(
            f"Sample shape length is expected to be {expected_dims}, actual length is {len(shape)}. Full incoming shape: {shape}"
        )


class TensorMetaMissingKey(Exception):
    def __init__(self, key: str, meta: dict):
        super().__init__(f"Key '{key}' missing from tensor meta '{str(meta)}'.")


class TensorDoesNotExistError(KeyError, AttributeError):
    def __init__(self, tensor_name: str):
        super().__init__(f"Tensor '{tensor_name}' does not exist.")


class TensorAlreadyExistsError(Exception):
    def __init__(self, key: str):
        super().__init__(
            f"Tensor '{key}' already exists. You can use the `exist_ok=True` parameter to ignore this error message."
        )


class TensorGroupDoesNotExistError(KeyError):
    def __init__(self, group_name: str):
        super().__init__(f"Tensor group '{group_name}' does not exist.")


class TensorGroupAlreadyExistsError(Exception):
    def __init__(self, key: str):
        super().__init__(
            f"Tensor group '{key}' already exists. A tensor group is created when a tensor has a '/' in its name, or using 'ds.create_group'."
        )


class InvalidTensorNameError(Exception):
    def __init__(self, name: str):
        if name:
            msg = (
                f"The use of a reserved attribute '{name}' as a tensor name is invalid."
            )
        else:
            msg = f"Tensor name cannot be empty."
        super().__init__(msg)


class InvalidTensorGroupNameError(Exception):
    def __init__(self, name: str):
        if name:
            msg = f"The use of a reserved attribute '{name}' as a tensor group name is invalid."
        else:
            msg = f"Tensor group name cannot be empty."
        super().__init__(msg)


class DynamicTensorNumpyError(Exception):
    def __init__(self, key: str, index, property_key: str):
        super().__init__(
            f"Tensor '{key}' with index = {str(index)} has dynamic '{property_key}' and cannot be converted into a `np.ndarray`. Try setting the parameter `aslist=True`"
        )


class InvalidShapeIntervalError(Exception):
    def __init__(
        self,
        message: str,
        lower: Optional[Sequence[int]] = None,
        upper: Optional[Sequence[int]] = None,
    ):
        s = message

        if lower is not None:
            s += f" lower={str(lower)}"

        if upper is not None:
            s += f" upper={str(upper)}"

        super().__init__(s)


class InvalidKeyTypeError(TypeError):
    def __init__(self, item: Any):
        super().__init__(
            f"Item '{str(item)}' of type '{type(item).__name__}' is not a valid key."
        )


class UnsupportedTensorTypeError(TypeError):
    def __init__(self, item: Any):
        super().__init__(
            f"Key of type '{type(item).__name__}' is not currently supported to convert to a tensor."
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


class UserNotLoggedInException(Exception):
    def __init__(self):
        message = (
            "You are not logged in and an API token was not found. To complete the operation, you can\n"
            "1. Login with your username and password using the `activeloop login` CLI command.\n"
            "2. Create an API token at https://app.activeloop.ai and use it in any of the following ways:\n"
            "    - Set the environment variable `ACTIVELOOP_TOKEN` to the token value.\n"
            "    - Use the CLI command `activeloop login -t <token>`.\n"
            "    - Pass the API token to the `token` parameter of this function.\n"
            "Visit https://docs.activeloop.ai/getting-started/using-activeloop-storage for more information."
        )
        super().__init__(message)


class InvalidHubPathException(Exception):
    def __init__(self, path):
        super().__init__(
            f"The Dataset's path is an invalid Deep Lake cloud path. It should be of the form hub://username/dataset got {path}."
        )


class PathNotEmptyException(Exception):
    def __init__(self, use_hub=True):
        if use_hub:
            super().__init__(
                f"Please use a url that points to an existing Deep Lake Dataset or an empty folder. If you wish to delete the folder and its contents, you may run deeplake.delete(dataset_path, force=True)."
            )
        else:
            super().__init__(
                f"Specified path is not empty. If you wish to delete the folder and its contents, you may run deeplake.delete(path, force=True)."
            )


# Exceptions encountered while interection with the Deep Lake backend
class AuthenticationException(Exception):
    def __init__(self, message="Authentication failed. Please try logging in again."):
        super().__init__(message)


class AuthorizationException(Exception):
    def __init__(
        self,
        message="You are not authorized to access this resource on Activeloop Server.",
        response=None,
    ):
        self.response = response
        super().__init__(message)


class InvalidPasswordException(AuthorizationException):
    def __init__(
        self,
        message="The password you provided was invalid.",
    ):
        super().__init__(message)


class CouldNotCreateNewDatasetException(AuthorizationException):
    def __init__(
        self,
        path: str,
    ):
        extra = ""
        if path.startswith("hub://"):
            extra = "Since the path is a `hub://` dataset, if you believe you should have write permissions, try running `activeloop login`."

        message = f"Dataset at '{path}' doesn't exist, and you have no permissions to create one there. Maybe a typo? {extra}"
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


class EmptyTokenException(Exception):
    def __init__(self, message="The authentication token is empty."):
        super().__init__(message)


# TODO Better S3 Exception handling


class S3Error(Exception):
    """Catchall for all errors encountered while working with S3"""


class S3GetError(S3Error):
    """Catchall for all errors encountered while working getting an object from S3"""


class S3SetError(S3Error):
    """Catchall for all errors encountered while working setting an object in S3"""


class S3DeletionError(S3Error):
    """Catchall for all errors encountered while working deleting an object in S3"""


class S3ListError(S3Error):
    """Catchall for all errors encountered while retrieving a list of objects present in S3"""


class S3GetAccessError(S3GetError):
    """Credentials related errors encountered while getting an object from S3"""


class CompressionError(Exception):
    pass


class UnsupportedCompressionError(CompressionError):
    def __init__(self, compression: Optional[str], htype: Optional[str] = None):
        if htype:
            super().__init__(
                f"Compression '{compression}' is not supported for {htype} htype."
            )
        else:
            super().__init__(
                f"Compression '{compression}' is not supported. Supported compressions: {deeplake.compressions}."
            )


class SampleCompressionError(CompressionError):
    def __init__(
        self,
        sample_shape: Tuple[int, ...],
        compression_format: Optional[str],
        message: str,
    ):
        super().__init__(
            f"Could not compress a sample with shape {str(sample_shape)} into '{compression_format}'. Raw error output: '{message}'.",
        )


class SampleDecompressionError(CompressionError):
    def __init__(self, path: Optional[str] = None):
        message = "Could not decompress sample"
        if path:
            message += f" at {path}"
        message += f". Either the sample's buffer is corrupted, or it is in an unsupported format. Supported compressions: {deeplake.compressions}."
        super().__init__(message)


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
            f"Invalid value '{value}' for tensor meta key '{key}'. {explanation}"
        )


class TensorMetaMissingRequiredValue(MetaError):
    def __init__(self, htype: str, key: Union[str, List[str]]):
        extra = ""
        if key == "sample_compression":
            extra = f"`sample_compression` may be `None` if you want your '{htype}' data to be uncompressed. Available compressors: {deeplake.compressions}"

        if isinstance(key, list):
            message = f"Htype '{htype}' requires you to specify either one of {key} inside the `create_tensor` method call. {extra}"
        else:
            message = f"Htype '{htype}' requires you to specify '{key}' inside the `create_tensor` method call. {extra}"
        super().__init__(message)


class TensorMetaInvalidHtypeOverwriteKey(MetaError):
    def __init__(self, htype: str, key: str, available_keys: Sequence[str]):
        super().__init__(
            f"Htype '{htype}' doesn't have a key for '{key}'. Available keys: {str(available_keys)}"
        )


class TensorDtypeMismatchError(MetaError):
    def __init__(self, expected: Union[np.dtype, str], actual: str, htype: str):
        msg = f"Dtype was expected to be '{expected}' instead it was '{actual}'. If you called `create_tensor` explicitly with `dtype`, your samples should also be of that dtype."

        # TODO: we may want to raise this error at the API level to determine if the user explicitly overwrote the `dtype` or not. (to make this error message more precise)
        # TODO: because if the user uses `dtype=np.uint8`, but the `htype` the tensor is created with has it's default dtype set as `uint8` also, then this message is ambiguous
        htype_dtype = deeplake.HTYPE_CONFIGURATIONS[htype].get("dtype", None)
        if htype_dtype is not None and htype_dtype == expected:
            msg += f" Htype '{htype}' expects samples to have dtype='{htype_dtype}'."
            super().__init__("")

        super().__init__(msg)


class InvalidTensorLinkError(MetaError):
    def __init__(self, msg="Invalid tensor link."):
        super().__init__(msg)


class TensorMetaMutuallyExclusiveKeysError(MetaError):
    def __init__(
        self, keys: Optional[List[str]] = None, custom_message: Optional[str] = None
    ):
        if custom_message:
            msg = custom_message
        else:
            msg = f"Following fields are mutually exclusive: {keys}. "
        super().__init__(msg)


class ReadOnlyModeError(Exception):
    def __init__(self, custom_message: Optional[str] = None):
        if custom_message is None:
            custom_message = "Modification when in read-only mode is not supported!"
        super().__init__(custom_message)


def is_primitive(sample):
    if isinstance(sample, (str, int, float, bool)):
        return True
    if isinstance(sample, dict):
        for x, y in sample.items():
            if not is_primitive(x) or not is_primitive(y):
                return False
        return True
    if isinstance(sample, (list, tuple)):
        for x in sample:
            if not is_primitive(x):
                return False
        return True
    return False


def get_truncated_sample(sample, max_half_len=50):
    if len(str(sample)) > max_half_len * 2:
        return (
            str(sample)[:max_half_len] + "..." + str(sample)[int(-max_half_len - 1) :]
        )
    return str(sample)


def has_path(sample):
    from deeplake.core.sample import Sample
    from deeplake.core.linked_sample import LinkedSample

    return isinstance(sample, LinkedSample) or (
        isinstance(sample, Sample) and sample.path is not None
    )


class TransformError(Exception):
    def __init__(self, index=None, sample=None, samples_processed=0, suggest=False):
        self.index = index
        self.sample = sample
        self.suggest = suggest
        # multiprocessing re raises error with str
        if isinstance(index, str):
            super().__init__(index)
        else:
            print_item = print_path = False
            if sample is not None:
                print_item = is_primitive(sample)
                print_path = has_path(sample)

            msg = f"Transform failed"
            if index is not None:
                msg += f" at index {index} of the input data"

            if print_item:
                msg += f" on the item: {get_truncated_sample(sample)}"
            elif print_path:
                msg += f" on the sample at path: '{sample.path}'"
            msg += "."

            if samples_processed > 0:
                msg += f" Last checkpoint: {samples_processed} samples processed. You can slice the input to resume from this point."

            msg += " See traceback for more details."

            if suggest:
                msg += (
                    " If you wish to skip the samples that cause errors,"
                    " please specify `ignore_errors=True`."
                )

            super().__init__(msg)


class SampleAppendError(Exception):
    def __init__(self, tensor, sample=None):
        print_item = print_path = False
        if sample is not None:
            print_item = is_primitive(sample)
            print_path = has_path(sample)
        if print_item or print_path:
            msg = "Failed to append the sample "

            if print_item:
                msg += str(sample) + " "
            elif print_path:
                msg += f"at path '{sample.path}' "
        else:
            msg = f"Failed to append a sample "

        msg += f"to the tensor '{tensor}'. See more details in the traceback."

        super().__init__(msg)


class SampleExtendError(Exception):
    def __init__(self, message):
        message += (
            " If you wish to skip the samples that cause errors,"
            " please specify `ignore_errors=True`."
        )
        super().__init__(message)


class FilterError(Exception):
    pass


class InvalidInputDataError(TransformError):
    def __init__(self, operation):
        super().__init__(
            f"The data_in to transform is invalid. It doesn't support {operation} operation. "
            "Please use a list, a Deep Lake dataset or an object that supports both __getitem__ and __len__. "
            "Generators are not supported."
        )


class UnsupportedSchedulerError(TransformError):
    def __init__(self, scheduler):
        super().__init__(
            f"Deep Lake compute currently doesn't support {scheduler} scheduler."
        )


class TensorMismatchError(TransformError):
    def __init__(self, tensors, output_keys, skip_ok=False):
        if skip_ok:
            super().__init__(
                f"One or more tensors generated during Deep Lake compute don't exist in the target dataset. With skip_ok=True, you can skip certain tensors in the transform, however you need to ensure that all tensors generated exist in the dataset.\n "
                f"Tensors in target dataset: {tensors}\n Tensors in output sample: {output_keys}"
            )
        else:
            super().__init__(
                f"One or more of the outputs generated during transform contain different tensors than the ones present in the target dataset of transform.\n "
                f"Tensors in target dataset: {tensors}\n Tensors in output sample: {output_keys}. If you want to do this, pass skip_ok=True to the eval method."
            )


class InvalidOutputDatasetError(TransformError):
    def __init__(
        self, message="The output Dataset to transform should not be `read_only`."
    ):
        super().__init__(message)


class InvalidTransformDataset(TransformError):
    def __init__(
        self,
        message="The TransformDataset (2nd argument to transform function) of one of the functions is invalid. All the tensors should have equal length for it to be valid.",
    ):
        super().__init__(message)


class HubComposeEmptyListError(TransformError):
    def __init__(self, message="Cannot deeplake.compose an empty list."):
        super().__init__(message)


class HubComposeIncompatibleFunction(TransformError):
    def __init__(self, index: int):
        super().__init__(
            f"The element passed to deeplake.compose at index {index} is incompatible. Ensure that functions are all decorated with deeplake.compute decorator and instead of passing my_fn, use my_fn() or my_fn(arg1=5, arg2=3) in the list."
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


class DatasetHandlerError(Exception):
    def __init__(self, message):
        super().__init__(message)


class MemoryDatasetCanNotBePickledError(Exception):
    def __init__(self):
        super().__init__(
            "Dataset having MemoryProvider as underlying storage should not be pickled as data won't be saved."
        )


class CorruptedSampleError(Exception):
    def __init__(self, compression, path: Optional[str] = None):
        message = f"Unable to decompress {compression} file"
        if path is not None:
            message += f" at {path}"
        message += "."
        super().__init__(message)


class VersionControlError(Exception):
    pass


class MergeError(Exception):
    pass


class MergeNotSupportedError(MergeError):
    def __init__(self):
        super().__init__(
            "This dataset was either created before merge functionality was added or id tensor wasn't created for one or more tensors. Create a new dataset to use merge."
        )


class MergeMismatchError(MergeError):
    def __init__(self, tensor_name, mismatch_type, original_value, target_value):
        message = f"Unable to merge, tensor {tensor_name} has different {mismatch_type}. Current:{original_value}, Target: {target_value}"
        super().__init__(message)


class MergeConflictError(MergeError):
    def __init__(self, conflict_tensors=None, message=""):
        if conflict_tensors:
            message = f"Unable to merge, tensors {conflict_tensors} have conflicts and conflict resolution argument was not provided. Use conflict_resolution='theirs' or conflict_resolution='ours' to resolve the conflict."
            super().__init__(message)
        else:
            super().__init__(message)


class CheckoutError(VersionControlError):
    pass


class CommitError(VersionControlError):
    pass


class EmptyCommitError(CommitError):
    pass


class TensorModifiedError(Exception):
    def __init__(self):
        super().__init__(
            "The target commit is not an ancestor of the current commit, modified can't be calculated."
        )


class GCSDefaultCredsNotFoundError(Exception):
    def __init__(self):
        super().__init__(
            "Unable to find default google application credentials at ~/.config/gcloud/application_default_credentials.json. "
            "Please make sure you initialized gcloud service earlier."
        )


class InvalidOperationError(Exception):
    def __init__(self, method: str, type: str):
        if method == "read_only":
            super().__init__("read_only property cannot be toggled for a dataset view.")
        else:
            super().__init__(f"{method} method cannot be called on a {type} view.")


class AgreementError(Exception):
    pass


class AgreementNotAcceptedError(AgreementError):
    def __init__(self, agreements=None):
        self.agreements = agreements
        super().__init__(
            "You did not accept the agreement. Make sure you type in the dataset name exactly as it appears."
        )


class NotLoggedInAgreementError(AgreementError):
    def __init__(self):
        super().__init__(
            "You are not logged in. Please log in to accept the agreement."
        )


class NotLoggedInError(AgreementError):
    def __init__(self, msg=None):
        msg = msg or (
            "This dataset includes an agreement that needs to be accepted before you can use it.\n"
            "You need to be signed in to accept this agreement.\n"
            "You can login using 'activeloop login' on the command line if you have an account or using 'activeloop register' if you don't have one."
        )

        super().__init__(msg)


class RenameError(Exception):
    def __init__(self, msg="Only name of the dataset can be different in new path."):
        super().__init__(msg)


class BufferError(Exception):
    pass


class InfoError(Exception):
    pass


class OutOfChunkCountError(Exception):
    pass


class OutOfSampleCountError(Exception):
    pass


class SampleHtypeMismatchError(Exception):
    def __init__(self, htype, sample_type):
        super().__init__(
            f"htype '{htype}' does not support samples of type {sample_type}."
        )


class EmptyTensorError(Exception):
    def __init__(self, message):
        super().__init__(message)


class DatasetViewSavingError(Exception):
    pass


class InvalidViewException(Exception):
    def __init__(self, message):
        super().__init__(message)


class ManagedCredentialsNotFoundError(Exception):
    def __init__(self, org_id, creds_key):
        super().__init__(
            f"Unable to find managed credentials '{creds_key}' for organization {org_id}."
        )


class UnableToReadFromUrlError(Exception):
    def __init__(self, url, status_code):
        super().__init__(f"Unable to read from url {url}. Status code: {status_code}")


class InvalidTokenException(Exception):
    def __init__(self):
        super().__init__(
            "Token is invalid. Make sure the full token string is included and try again."
        )


class TokenPermissionError(Exception):
    def __init__(self, message=None):
        if message is None:
            message = (
                "A dataset does not exist at the specified path, or you do not have "
                "sufficient permissions to load or create one. Please check the dataset "
                "path and make sure that you have sufficient permissions to the path."
            )
        super().__init__(message)


class SampleAppendingError(Exception):
    def __init__(self):
        message = """Cannot append sample because tensor(s) are not specified. Expected input to ds.append is a dictionary. To append samples, you need to either specify the tensors and append the samples as a dictionary, like: `ds.append({"image_tensor": sample, "label_tensor": sample})` or you need to call `append` method of the required tensor, like: `ds.image_tensor.append(sample)`"""
        super().__init__(message)


class SampleExtendingError(Exception):
    def __init__(self):
        message = (
            "Cannot extend because tensor(s) are not specified. Expected input to ds.extend is a dictionary. "
            "To extend tensors, you need to either specify the tensors and add the samples as a dictionary, like: "
            "`ds.extend({'image_tensor': samples, 'label_tensor': samples})` or you need to call `extend` method of the required tensor, "
            "like: `ds.image_tensor.extend(samples)`"
        )


class DatasetTooLargeToDelete(Exception):
    def __init__(self, ds_path):
        message = f"Deep Lake Dataset {ds_path} was too large to delete. Try again with large_ok=True."
        super().__init__(message)


class TensorTooLargeToDelete(Exception):
    def __init__(self, tensor_name):
        message = f"Tensor {tensor_name} was too large to delete. Try again with large_ok=True."
        super().__init__(message)


class BadLinkError(Exception):
    def __init__(self, link, creds_key):
        message = f"Verification of link failed. Make sure that the link you are trying to append is correct.\n\nFailed link: {link}\ncreds_key used: {creds_key}"
        if creds_key is None:
            extra = """\n\nNo credentials have been provided to access the link. If the link is not publibly accessible, add access credentials to your dataset\
 and use the appropriate creds_key."""
        else:
            extra = """\n\nPlease also check that the specified credentials have access to the link."""
        message += extra
        super().__init__(message)


class IngestionError(Exception):
    pass


class DatasetConnectError(Exception):
    pass


class InvalidSourcePathError(DatasetConnectError):
    pass


class InvalidDestinationPathError(DatasetConnectError):
    pass


class UnprocessableEntityException(Exception):
    def __init__(self, message: Optional[str] = None) -> None:
        if message is None or message == " ":
            message = "Some request parameters were invalid."


class GroupInfoNotSupportedError(Exception):
    def __init__(self):
        message = "Tensor groups does not have info attribute. Please use `dataset.info` or `dataset.tensor.info`."
        super().__init__(message)


class InvalidDatasetNameException(Exception):
    def __init__(self, path_type):
        if path_type == "local":
            message = "Local dataset names can only contain letters, numbers, spaces, `-`, `_` and `.`."
        else:
            message = "Please specify a dataset name that contains only letters, numbers, hyphens and underscores."
        super().__init__(message)


class UnsupportedParameterException(Exception):
    pass


class UnsupportedExtensionError(Exception):
    def __init__(self, extension, htype=""):
        if htype:
            htype = f"For {htype} htype "
        message = f"{htype}{extension} is not supported"

        super().__init__(message)


class DatasetCorruptError(Exception):
    def __init__(self, message, action="", cause=None):
        self.message = message
        self.action = action
        self.__cause__ = cause

        super().__init__(self.message + (" " + self.action if self.action else ""))


class SampleReadError(Exception):
    def __init__(self, path: str):
        super().__init__(f"Unable to read sample from {path}")


class GetChunkError(Exception):
    def __init__(
        self,
        chunk_key: Optional[str],
        global_index: Optional[int] = None,
        tensor_name: Optional[str] = None,
    ):
        self.chunk_key = chunk_key
        message = "Unable to get chunk"
        if chunk_key is not None:
            message += f" '{chunk_key}'"
        if global_index is not None:
            message += f" while retrieving data at index {global_index}"
        if tensor_name is not None:
            message += f" in tensor {tensor_name}"
        message += "."
        super().__init__(message)


class ReadSampleFromChunkError(Exception):
    def __init__(
        self,
        chunk_key: Optional[str],
        global_index: Optional[int] = None,
        tensor_name: Optional[str] = None,
    ):
        self.chunk_key = chunk_key
        message = "Unable to read sample"
        if global_index is not None:
            message += f" at index {global_index}"
        message += " from chunk"
        if chunk_key is not None:
            message += f" '{chunk_key}'"
        if tensor_name is not None:
            message += f" in tensor {tensor_name}"
        message += "."
        super().__init__(message)


class GetDataFromLinkError(Exception):
    def __init__(
        self,
        link: str,
        global_index: Optional[int] = None,
        tensor_name: Optional[str] = None,
    ):
        self.link = link
        message = f"Unable to get data from link {link}"
        if global_index is not None:
            message += f" while retrieving data at index {global_index}"
        if tensor_name is not None:
            message += f" in tensor {tensor_name}"
        message += "."
        super().__init__(message)


class TransformFailedError(Exception):
    def __init__(self, global_index):
        super().__init__(
            f"Dataloader transform failed while processing sample at index {global_index}."
        )


class MissingCredsError(Exception):
    pass


class EmptyPolygonError(Exception):
    pass


class MissingManagedCredsError(Exception):
    pass


class SampleUpdateError(Exception):
    def __init__(self, key: str):
        super().__init__(f"Unable to update sample in tensor {key}.")


class AllSamplesSkippedError(Exception):
    def __init__(self):
        super().__init__(
            "All samples were skipped during the transform. "
            "Ensure your transform pipeline is correct before you set `ignore_errors=True`."
        )


class FailedIngestionError(Exception):
    pass


class IncorrectEmbeddingShapeError(Exception):
    def __init__(self):
        super().__init__(
            "The embedding function returned embeddings of different shapes. "
            "Please either use different embedding function or exclude invalid "
            "files that are not supported by the embedding function. "
        )


class IncompatibleHtypeError(Exception):
    def __init__(self, msg):
        super().__init__(msg)
