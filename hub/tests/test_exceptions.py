from hub.exceptions import (
    HubException,
    AuthenticationException,
    AuthorizationException,
    NotFound,
    NotFoundException,
    BadRequestException,
    OverLimitException,
    ServerException,
    BadGatewayException,
    GatewayTimeoutException,
    WaitTimeoutException,
    LockedException,
    HubDatasetNotFoundException,
    PermissionException,
    ShapeArgumentNotFoundException,
    SchemaArgumentNotFoundException,
    ValueShapeError,
    NoneValueException,
    ShapeLengthException,
    ModuleNotInstalledException,
    WrongUsernameException,
    NotHubDatasetToAppendException,
    NotHubDatasetToOverwriteException,
    DynamicTensorNotFoundException,
    DynamicTensorShapeException,
)


class Response:
    def __init__(self, noerror=False):
        self.content = "Hello World"
        if noerror:
            self.json = lambda: {"message": "Hello There"}


def test_exceptions():
    HubException()
    AuthenticationException()
    AuthorizationException(Response())
    AuthorizationException(Response(noerror=True))
    NotFoundException()
    BadRequestException(Response())
    BadRequestException(Response(noerror=True))
    OverLimitException()
    ServerException()
    BadGatewayException()
    GatewayTimeoutException()
    WaitTimeoutException()
    LockedException()
    HubDatasetNotFoundException("Hello")
    PermissionException("Hello")
    ShapeLengthException()
    ShapeArgumentNotFoundException()
    SchemaArgumentNotFoundException()
    ValueShapeError("Shape 1", "Shape 2")
    NoneValueException("Yahoo!")
    ModuleNotInstalledException("my_module")
    WrongUsernameException("usernameX")
    NotHubDatasetToOverwriteException()
    NotHubDatasetToAppendException()
    DynamicTensorNotFoundException()

    DynamicTensorShapeException("none")
    DynamicTensorShapeException("length")
    DynamicTensorShapeException("not_equal")
    DynamicTensorShapeException("another_cause")
