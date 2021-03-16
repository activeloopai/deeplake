"""
License:
This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

from hub.exceptions import (
    DaskModuleNotInstalledException,
    HubException,
    AuthenticationException,
    AuthorizationException,
    NotFound,
    NotFoundException,
    BadRequestException,
    NotIterable,
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
    NotIterable()
    DaskModuleNotInstalledException()
    DynamicTensorShapeException("none")
    DynamicTensorShapeException("length")
    DynamicTensorShapeException("not_equal")
    DynamicTensorShapeException("another_cause")
    NotFound()
