class EmptyTokenException(Exception):
    def __init__(self, message="The authentication token is empty."):
        super().__init__(message)


class ValidationDatasetMissingError(Exception):
    def __init__(self):
        msg = (
            "Validation dataset is not specified even though validate = True. "
            "Please set validate = False or specify a validation dataset."
        )
        super().__init__(msg)


class InvalidImageError(Exception):
    def __init__(self, column_name, ex):
        msg = f"Error on {column_name} data getting: {str(ex)}"
        super().__init__(msg)


class InvalidSegmentError(Exception):
    def __init__(self, column_name, ex):
        msg = f"Error on {column_name} data getting: {str(ex)}"
        super().__init__(msg)
