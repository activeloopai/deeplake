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

    return start, stop, step, reverse
