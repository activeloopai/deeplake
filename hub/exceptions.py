
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