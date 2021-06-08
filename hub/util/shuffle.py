import numpy as np


def shuffle(ds):
    """Returns a shuffled copy of a given Dataset."""
    idxs = np.arange(len(ds))
    np.random.shuffle(idxs)
    return ds[list(idxs)]
