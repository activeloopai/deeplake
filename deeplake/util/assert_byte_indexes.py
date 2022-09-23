from hub.util.exceptions import InvalidBytesRequestedError


def assert_byte_indexes(start_byte, end_byte):
    """Checks whether the bytes are valid.

    Args:
        start_byte (int): The starting index to be checked.
        end_byte (int): The end index to be checked.

    Raises:
        InvalidBytesRequestedError: If `start_byte` > `end_byte` or `start_byte` < 0 or `end_byte` < 0
    """
    start_byte = start_byte or 0
    if start_byte < 0:
        raise InvalidBytesRequestedError()
    if end_byte is not None and (start_byte > end_byte or end_byte < 0):
        raise InvalidBytesRequestedError()
