"""
License:
This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

from click import ClickException

from hub.report import (
    hub_reporter,
    ExceptionWithReporting,
    hub_tags,
)


class OutOfBoundsError(ExceptionWithReporting):
    """Raised upon finding a missing chunk."""

    pass


class AlignmentError(ExceptionWithReporting):
    """Raised when there is an Alignment error."""

    pass


class IncompatibleShapes(ExceptionWithReporting):
    """Shapes do not match"""

    pass


class IncompatibleBroadcasting(ExceptionWithReporting):
    """Broadcasting issue"""

    pass


class IncompatibleTypes(ExceptionWithReporting):
    """Types can not cast"""

    pass


class WrongTypeError(ExceptionWithReporting):
    """Types is not supported"""

    pass


class NotAuthorized(ExceptionWithReporting):
    """Types is not supported"""

    pass


class NotFound(ExceptionWithReporting):
    """When Info could not be found for array"""

    pass


class FileSystemException(ExceptionWithReporting):
    """Error working with local file system"""

    pass


class S3Exception(ExceptionWithReporting):
    """Error working with AWS"""

    pass


class S3CredsParseException(ExceptionWithReporting):
    """Can't parse AWS creds"""

    pass


class HubException(ClickException):
    def __init__(self, message=None, code=None):
        super().__init__(message)


class AuthenticationException(HubException):
    def __init__(self, message="Authentication failed. Please login again."):
        super().__init__(message=message)
        hub_reporter.error_report(self, tags=hub_tags)


class AuthorizationException(HubException):
    def __init__(self, response):
        try:
            message = response.json()["message"]
        except (KeyError, AttributeError):
            message = "You are not authorized to access this resource on Snark AI."
        super().__init__(message=message)
        hub_reporter.error_report(self, tags=hub_tags)


class NotFoundException(HubException):
    def __init__(
        self,
        message="The resource you are looking for was not found. Check if the name or id is correct.",
    ):
        super().__init__(message=message)
        hub_reporter.error_report(self, tags=hub_tags)


class BadRequestException(HubException):
    def __init__(self, response):
        try:
            message = "One or more request parameters is incorrect\n%s" % str(
                response.json()["message"]
            )
        except (KeyError, AttributeError):
            message = "One or more request parameters is incorrect, %s" % str(
                response.content
            )
        super().__init__(message=message)
        hub_reporter.error_report(self, tags=hub_tags)


class OverLimitException(HubException):
    def __init__(
        self,
        message="You are over the allowed limits for this operation. Consider upgrading your account.",
    ):
        super().__init__(message=message)
        hub_reporter.error_report(self, tags=hub_tags)


class ServerException(HubException):
    def __init__(self, message="Internal Snark AI server error."):
        super().__init__(message=message)
        hub_reporter.error_report(self, tags=hub_tags)


class BadGatewayException(HubException):
    def __init__(self, message="Invalid response from Snark AI server."):
        super().__init__(message=message)
        hub_reporter.error_report(self, tags=hub_tags)


class GatewayTimeoutException(HubException):
    def __init__(self, message="Snark AI server took too long to respond."):
        super().__init__(message=message)
        hub_reporter.error_report(self, tags=hub_tags)


class WaitTimeoutException(HubException):
    def __init__(self, message="Timeout waiting for server state update."):
        super().__init__(message=message)
        hub_reporter.error_report(self, tags=hub_tags)


class LockedException(HubException):
    def __init__(self, message="Resource locked."):
        super().__init__(message=message)
        hub_reporter.error_report(self, tags=hub_tags)


class HubDatasetNotFoundException(HubException):
    def __init__(self, response):
        message = f"The dataset with tag {response} was not found"
        super(HubDatasetNotFoundException, self).__init__(message=message)
        hub_reporter.error_report(self, tags=hub_tags)


class PermissionException(HubException):
    def __init__(self, response):
        message = f"No permision to store the dataset at {response}"
        super(PermissionException, self).__init__(message=message)
        hub_reporter.error_report(self, tags=hub_tags)


class ShapeArgumentNotFoundException(HubException):
    def __init__(self):
        message = "Parameter 'shape' should be provided for Dataset creation."
        super(HubException, self).__init__(message=message)
        hub_reporter.error_report(self, tags=hub_tags)


class DirectoryNotEmptyException(HubException):
    def __init__(self, dst_url):
        message = f"The destination url {dst_url} for copying dataset is not empty. Delete the directory manually or use Dataset.delete if it's a Hub dataset"
        super(HubException, self).__init__(message=message)
        hub_reporter.error_report(self, tags=hub_tags)


class SchemaArgumentNotFoundException(HubException):
    def __init__(self):
        message = "Parameter 'schema' should be provided for Dataset creation."
        super(HubException, self).__init__(message=message)
        hub_reporter.error_report(self, tags=hub_tags)


class InvalidVersionInfoException(HubException):
    def __init__(self):
        message = "The pickle file with version info exists but the version info inside is invalid. Proceeding without version control."
        super(HubException, self).__init__(message=message)


class ValueShapeError(HubException):
    def __init__(self, correct_shape, wrong_shape):
        message = f"parameter 'value': expected array with shape {correct_shape}, got {wrong_shape}"
        super(HubException, self).__init__(message=message)
        hub_reporter.error_report(self, tags=hub_tags)


class NoneValueException(HubException):
    def __init__(self, param):
        message = f"Parameter '{param}' should be provided"
        super(HubException, self).__init__(message=message)
        hub_reporter.error_report(self, tags=hub_tags)


class ShapeLengthException(HubException):
    def __init__(self):
        message = "Parameter 'shape' should be a tuple of length 1"
        super(HubException, self).__init__(message=message)
        hub_reporter.error_report(self, tags=hub_tags)


class ModuleNotInstalledException(HubException):
    def __init__(self, module_name):
        message = f"Module '{module_name}' should be installed to convert the Dataset to the {module_name} format"
        super(HubException, self).__init__(message=message)
        hub_reporter.error_report(self, tags=hub_tags)


class ReadModeException(HubException):
    def __init__(self, method_name):
        message = f"Can't call {method_name} as the dataset is in read mode"
        super(HubException, self).__init__(message=message)


class VersioningNotSupportedException(HubException):
    def __init__(self, method_name):
        message = f"This dataset was created before version control, it does not support {method_name} functionality."
        super(HubException, self).__init__(message=message)


class DaskModuleNotInstalledException(HubException):
    def __init__(self, message=""):
        message = "Dask has been deprecated and made optional. Older versions of 0.x hub datasets require loading dask. Please install it: pip install 'dask[complete]>=2.30'"
        super(HubException, self).__init__(message=message)
        hub_reporter.error_report(self, tags=hub_tags)


class WrongUsernameException(HubException):
    def __init__(self, username):
        message = (
            f"Username {username} doesn't have access to the given url. Please check your permissions "
            "or make sure that the username provided in the url matches the one used during login or ."
        )
        super(HubException, self).__init__(message=message)
        hub_reporter.error_report(self, tags=hub_tags)


class NotHubDatasetToOverwriteException(HubException):
    def __init__(self):
        message = (
            "Unable to overwrite the dataset. "
            "The provided directory is not empty and doesn't contain information about any Hub Dataset. "
            "This is a safety check so it won't be possible to overwrite (delete) any folder other than Dataset folder. "
            "If this error persists in case of Dataset folder then it means your Dataset data is corrupted. "
            "In that case feel free to create an issue in here https://github.com/activeloopai/Hub"
        )
        super(HubException, self).__init__(message=message)
        hub_reporter.error_report(self, tags=hub_tags)


class NotHubDatasetToAppendException(HubException):
    def __init__(self):
        message = (
            "Unable to append to the dataset. "
            "The provided directory is not empty and doesn't contain information about any Hub Dataset "
        )
        super(HubException, self).__init__(message=message)
        hub_reporter.error_report(self, tags=hub_tags)


class DynamicTensorNotFoundException(HubException):
    def __init__(self):
        message = "Unable to find dynamic tensor"
        super(HubException, self).__init__(message=message)
        hub_reporter.error_report(self, tags=hub_tags)


class DynamicTensorShapeException(HubException):
    def __init__(self, exc_type):
        if exc_type == "none":
            message = "Parameter 'max_shape' shouldn't contain any 'None' value"
        elif exc_type == "length":
            message = "Lengths of 'shape' and 'max_shape' should be equal"
        elif exc_type == "not_equal":
            message = "All not-None values from 'shape' should be equal to the corresponding values in 'max_shape'"
        else:
            message = "Wrong 'shape' or 'max_shape' values"
        super(HubException, self).__init__(message=message)
        hub_reporter.error_report(self, tags=hub_tags)


class NotIterable(HubException):
    def __init__(self):
        message = "First argument to transform function should be iterable"
        super(HubException, self).__init__(message=message)
        hub_reporter.error_report(self, tags=hub_tags)


class AddressNotFound(HubException):
    def __init__(self, address):
        message = f"The address {address} does not refer to any existing branch or commit id. use create=True to create a new branch with this address"
        super(HubException, self).__init__(message=message)
        hub_reporter.error_report(self, tags=hub_tags)


class NotZarrFolderException(ExceptionWithReporting):
    pass


class StorageTensorNotFoundException(ExceptionWithReporting):
    pass
