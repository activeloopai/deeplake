def slice_split_tuple(slice_):
    path = ""
    list_slice = []
    for sl in slice_:
        if isinstance(sl, str):
            path += sl if sl.startswith("/") else "/" + sl
        elif isinstance(sl, int) or isinstance(sl, slice):
            list_slice.append(sl)
        else:
            raise TypeError("type {} isn't supported in dataset slicing".format(type(sl)))
    tuple_slice = tuple(list_slice)
    if path == "":
        raise ValueError("No path found in slice!")
    return path, tuple_slice


def slice_extract_info(slice_, num):
    assert isinstance(slice_, slice)

    if slice_.step is not None and slice_.step < 0:       # negative step not supported
        raise ValueError("Negative step not supported in dataset slicing")

    offset = 0

    if slice_.start is not None:
        if slice_.start < 0:                 # make indices positive if possible
            slice_.start += num
            if slice_.start < 0:
                raise IndexError('index out of bounds for dimension with length {}'.format(num))

        if slice_.start >= num:
            raise IndexError('index out of bounds for dimension with length {}'.format(num))
        offset = slice_.start

    if slice_.stop is not None:
        if slice_.stop < 0:                   # make indices positive if possible
            slice_.stop += num
            if slice_.stop < 0:
                raise IndexError('index out of bounds for dimension with length {}'.format(num))

        if slice_.stop > num:
            raise IndexError('index out of bounds for dimension with length {}'.format(num))

    if slice_.start is not None and slice_.stop is not None:
        if slice_.stop < slice_.start:    # return empty
            num = 0
        else:
            num = slice_.stop - slice_.start
    elif slice_.start is None and slice_.stop is not None:
        num = slice_.stop
    elif slice_.start is not None and slice_.stop is None:
        num = num - slice_.start
    else:
        num = num

    return num, offset


def combine(slice_, num=None, ofs=0):
    if isinstance(slice_, int):
        if num is not None and slice_ >= num:
            raise IndexError('index out of bounds for dimension with length {}'.format(num))
        return ofs + slice_
    elif isinstance(slice_, slice):
        if slice_.step is not None and slice_.step < 0:       # negative step not supported
            raise ValueError("Negative step not supported in dataset slicing")
        if slice_.start is None and slice_.stop is None:
            if num is None:
                return slice(ofs, None)
            else:
                return slice(ofs, ofs + num)
        elif slice_.start is not None and slice_.stop is None:
            if num is None:
                return slice(ofs + slice_.start, None)
            else:
                if slice_.start >= num:
                    raise IndexError('index out of bounds for dimension with length {}'.format(num))
                return slice(ofs + slice_.start, ofs + num)
        elif slice_.start is None and slice_.stop is not None:
            if num is None:
                return slice(ofs, ofs + slice_.stop)
            else:
                if slice_.stop > num:
                    raise IndexError('index out of bounds for dimension with length {}'.format(num))
                return slice(ofs, ofs + slice_.stop)
        else:
            if slice_.start > slice_.stop:
                raise IndexError('start index is greater than stop index')
            if num is None:
                return slice(ofs + slice_.start, ofs + slice_.stop)
            else:
                if slice_.stop > num:
                    raise IndexError('index out of bounds for dimension with length {}'.format(num))
                return slice(ofs + slice_.start, ofs + slice_.stop)
    else:
        raise TypeError("type {} isn't supported in dataset slicing".format(type(slice_)))
