def normalize_index(index, nframes):
    reverse = False
    if index is None:
        start = 0
        stop = nframes
        step = 1
    elif isinstance(index, int):
        if index >= 0:
            start = index
        else:
            start = nframes + index
        stop = start + 1
        step = 1
    elif isinstance(index, slice):
        step = index.step
        if step is None:
            step = 1
        elif step < 0:
            step = abs(step)
            reverse = True

        start = index.start
        if start is None:
            start = 0 if not reverse else nframes
        elif start < 0:
            start = nframes + start

        stop = index.stop
        if stop is None:
            stop = nframes if not reverse else -1
        elif stop < 0:
            stop = nframes + stop

        if reverse:
            start, stop = stop + 1, start + 1
    elif isinstance(index, list):
        raise IndexError(
            "Cannot specify a list video frames. You must specify a range with an optional step such as [5:10] or [0:100:5]"
        )
    else:
        raise IndexError(
            f"Invalid video index type: {type(index)}. You must specify either a specific frame index or a range."
        )

    return start, stop, step, reverse
