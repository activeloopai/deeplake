def slice_split(slice_):
    """Splits a slice into subpath and list of slices"""
    path = ""
    list_slice = []
    for sl in slice_:
        if isinstance(sl, str):
            path += sl if sl.startswith("/") else "/" + sl
        elif isinstance(sl, int) or isinstance(sl, slice):
            list_slice.append(sl)
        else:
            raise TypeError(
                "type {} isn't supported in dataset slicing".format(type(sl))
            )
    return path, list_slice


def slice_extract_info(slice_, num):
    """Extracts number of samples and offset from slice"""
    if isinstance(slice_, int):
        slice_ = slice_ + num if slice_ < 0 else slice_
        if slice_ >= num or slice_ < 0:
            raise IndexError(
                "index out of bounds for dimension with length {}".format(num)
            )
        return (1, slice_)

    if slice_.step is not None and slice_.step < 0:  # negative step not supported
        raise ValueError("Negative step not supported in dataset slicing")
    offset = 0
    if slice_.start is not None:
        slice_ = (
            slice(slice_.start + num, slice_.stop) if slice_.start < 0 else slice_
        )  # make indices positive if possible
        if slice_.start < 0 or slice_.start >= num:
            raise IndexError(
                "index out of bounds for dimension with length {}".format(num)
            )
        offset = slice_.start
    if slice_.stop is not None:
        slice_ = (
            slice(slice_.start, slice_.stop + num) if slice_.stop < 0 else slice_
        )  # make indices positive if possible
        if slice_.stop < 0 or slice_.stop > num:
            raise IndexError(
                "index out of bounds for dimension with length {}".format(num)
            )
    if slice_.start is not None and slice_.stop is not None:
        num = 0 if slice_.stop < slice_.start else slice_.stop - slice_.start
    elif slice_.start is None and slice_.stop is not None:
        num = slice_.stop
    elif slice_.start is not None and slice_.stop is None:
        num = num - slice_.start
    return num, offset
