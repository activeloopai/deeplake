import numpy as np


def is_empty(sample):
    return (
        isinstance(sample, list)
        and len(sample) == 0
        or isinstance(sample, np.ndarray)
        and sample.shape == ()
    )
