import numpy as np


def shuffle(ds):
    """Returns a shuffled wrapper of a given Dataset."""
    idxs = np.arange(len(ds))
    np.random.shuffle(idxs)
    return ds[idxs.tolist()]
