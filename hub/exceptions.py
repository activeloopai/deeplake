from click import ClickException


class OutOfBoundsError(Exception):
    """Raised upon finding a missing chunk."""

    pass


class AlignmentError(Exception):
    """Raised when there is an Alignment error."""

    pass


class IncompatibleShapes(Exception):
    """Shapes do not match"""

    pass


class IncompatibleBroadcasting(Exception):
    """Broadcasting issue"""

    pass


class IncompatibleTypes(Exception):
    """Types can not cast"""

    pass


class WrongTypeError(Exception):
    """Types is not supported"""

    pass


class NotAuthorized(Exception):
    """Types is not supported"""

    pass


class NotFound(Exception):
    """When Info could not be found for array"""

    pass


class FileSystemException(Exception):
    """Error working with local file system"""

    pass


class S3Exception(Exception):
    """Error working with AWS"""

    pass


class S3CredsParseException(Exception):
    """Can't parse AWS creds"""

    pass


class HubException(ClickException):
    def __init__(self, message=None, code=None):
        super(HubException, self).__init__(message)


class AuthenticationException(HubException):
    def __init__(self, message="Authentication failed. Please login again."):
        super(AuthenticationException, self).__init__(message=message)


class AuthorizationException(HubException):
    def __init__(self, response):
        try:
            message = response.json()["message"]
        except (KeyError, AttributeError):
            message = "You are not authorized to access this resource on Snark AI."
        super(AuthorizationException, self).__init__(message=message)


class NotFoundException(HubException):
    def __init__(
        self,
        message="The resource you are looking for was not found. Check if the name or id is correct.",
    ):
        super(NotFoundException, self).__init__(message=message)


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
        super(BadRequestException, self).__init__(message=message)


class OverLimitException(HubException):
    def __init__(
        self,
        message="You are over the allowed limits for this operation. Consider upgrading your account.",
    ):
        super(OverLimitException, self).__init__(message=message)


class ServerException(HubException):
    def __init__(self, message="Internal Snark AI server error."):
        super(ServerException, self).__init__(message=message)


class BadGatewayException(HubException):
    def __init__(self, message="Invalid response from Snark AI server."):
        super(BadGatewayException, self).__init__(message=message)


class GatewayTimeoutException(HubException):
    def __init__(self, message="Snark AI server took too long to respond."):
        super(GatewayTimeoutException, self).__init__(message=message)


class WaitTimeoutException(HubException):
    def __init__(self, message="Timeout waiting for server state update."):
        super(WaitTimeoutException, self).__init__(message=message)


class LockedException(HubException):
    def __init__(self, message="Resource locked."):
        super(LockedException, self).__init__(message=message)


class HubDatasetNotFoundException(HubException):
    def __init__(self, response):
        message = f"The dataset with tag {response} was not found"
        super(HubDatasetNotFoundException, self).__init__(message=message)


class PermissionException(HubException):
    def __init__(self, response):
        message = f"No permision to store the dataset at {response}"
        super(PermissionException, self).__init__(message=message)

class ShapeArgumentNotFoundException(HubException):
    def __init__(self):
        message = f"Parameter 'shape' should be provided for Dataset creation."
        super(HubException, self).__init__(message=message)

class SchemaArgumentNotFoundException(HubException):
    def __init__(self):
        message = f"Parameter 'schema' should be provided for Dataset creation."
        super(HubException, self).__init__(message=message)

class ValueShapeError(HubException):
    def __init__(self, correct_shape, wrong_shape):
        message = f"parameter 'value': expected array with shape {correct_shape}, got {wrong_shape}"
        super(HubException, self).__init__(message=message)

class ModuleNotInstalledException(HubException):
    def __init__(self, module_name):
        message = f"Module '{module_name}' should be installed to convert the Dataset to the {module_name} format"
        super(HubException, self).__init__(message=message)

class WrongUsernameException(HubException):
    def __init__(self, username):
        message = f"The username {username} was not found. Make sure that the username provided in the url " \
                   "matches the one used during login."
        super(HubException, self).__init__(message=message)

class NotHubDatasetToOverwriteException(HubException):
    def __init__(self):
        message = "Unable to overwrite the dataset. "  \
                "The provided directory is not empty and doesn't contain information about any Hub Dataset "
        super(HubException, self).__init__(message=message)

class NotHubDatasetToAppendException(HubException):
    def __init__(self):
        message = "Unable to append to the dataset. "  \
                  "The provided directory is not empty and doesn't contain information about any Hub Dataset "
        super(HubException, self).__init__(message=message)


class NotZarrFolderException(Exception):
    pass


class StorageTensorNotFoundException(Exception):
    pass


class DynamicTensorNotFoundException(Exception):
    pass