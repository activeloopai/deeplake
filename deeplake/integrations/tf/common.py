from deeplake.util.iterable_ordered_dict import IterableOrderedDict
from deeplake.core.polygon import Polygons
import numpy as np


def collate_fn(batch):
    elem = batch[0]
    if isinstance(elem, IterableOrderedDict):
        return IterableOrderedDict(
            (key, collate_fn([d[key] for d in batch])) for key in elem.keys()
        )

    if isinstance(elem, np.ndarray) and elem.size > 0 and isinstance(elem[0], str):
        batch = [it[0] for it in batch]
    elif isinstance(elem, Polygons):
        batch = [it.numpy() for it in batch]
    return default_collate(batch)


def default_collate(batch):
    import tensorflow as tf

    v = batch[0]
    if isinstance(v, np.ndarray):
        return np.stack(batch)
    elif isinstance(v, tf.Tensor):
        return tf.stack(batch)
    else:
        return np.array(batch)
