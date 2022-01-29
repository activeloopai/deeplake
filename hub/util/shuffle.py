from numpy import (
    arange as np_arange,
    random as np_random
)


def shuffle(ds):
    """Returns a shuffled wrapper of a given Dataset."""
    idxs = np_arange(len(ds))
    np_random.shuffle(idxs)
    return ds[idxs.tolist()]
