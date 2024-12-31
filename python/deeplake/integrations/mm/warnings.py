import warnings


def always_warn(*args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        warnings.warn(*args, **kwargs)
