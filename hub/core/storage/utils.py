def check_byte_indexes(start_byte, end_byte):
    """
    Checks the bytes passed to it, to see if they are valid.
    Invalid cases are when start_byte or end_byte are less than 0 or if start_byte>end_byte

    Args:
        start_byte (int): The starting index to be checked
        end_byte (int): The end index to be checked

    Returns:
        None

    Raises:
        Exception #TODO Proper
    """
    # we don't allow negative indexes while accessing bytes
    start_byte = start_byte or 0
    if start_byte < 0:
        raise Exception  # TODO add proper exception
    if end_byte is not None and (start_byte > end_byte or end_byte < 0):
        raise Exception  # TODO add proper exception
