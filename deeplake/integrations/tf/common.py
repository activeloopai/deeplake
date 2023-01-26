from deeplake.util.iterable_ordered_dict import IterableOrderedDict
from deeplake.core.polygon import Polygons
import numpy as np


def collate_fn(batch):
    from tensorflow import convert_to_tensor

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
        stack = np.stack(batch)
        return tf.convert_to_tensor(stack)
    elif isinstance(v, tf.Tensor):
        return tf.stack(batch)
    elif isinstance(v, (tuple, list)):
        ls = [default_collate([b[i] for b in batch]) for i in range(len(v))]
        if isinstance(v, tuple):
            return tuple(ls)
        return ls
    elif isinstance(v, dict):
        return {key: default_collate([d[key] for d in batch]) for key in v}
    else:
        stack = np.array(batch)
        return tf.convert_to_tensor(stack)
