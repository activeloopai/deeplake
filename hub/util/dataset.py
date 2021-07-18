from hub.util.exceptions import ReadOnlyModeError


def try_flushing(ds):
    try:
        ds.flush()
    except ReadOnlyModeError:
        pass
