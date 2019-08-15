
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

class ArrayNotFound(Exception):
    """When Info could not be found for array"""
    pass 