import numpy as np
from deeplake.core.seed import DeeplakeRandom


def shuffle(ds):
    """Returns a shuffled wrapper of a given Dataset."""
    prev_state = np.random.get_state()
    np.random.seed(DeeplakeRandom().get_seed())
    idxs = np.arange(len(ds))
    np.random.shuffle(idxs)
    np.random.set_state(prev_state)
    return ds[idxs.tolist()]
