def merge_slices(existing_slice: slice, new_slice: slice) -> slice:
    """Compose two slice objects

    Given an iterable x, the following should be equivalent:
    x[existing_slice][new_slice]
    x[merge_slices(existing_slice, new_slice)]

    Args:
        existing_slice (slice): The existing slice to be restricted.
        new_slice (slice): The new slice to be applied to the existing slice.

    Returns:
        slice: the composition of the given slices
    """
    return existing_slice  # TODO: implement this
