from hub.util.iterable_ordered_dict import IterableOrderedDict
import numpy as np


def collate_fn(batch):
    import torch

    elem = batch[0]
    if elem and isinstance(elem, np.ndarray) and isinstance(elem[0], str):
        batch = [it[0] for it in batch]
    if isinstance(elem, IterableOrderedDict):
        return IterableOrderedDict(
            (key, collate_fn([d[key] for d in batch])) for key in elem.keys()
        )
    return torch.utils.data._utils.collate.default_collate(batch)


def convert_fn(data):
    import torch

    if data and isinstance(data, np.ndarray) and isinstance(data[0], str):
        data = data[0]
    if isinstance(data, IterableOrderedDict):
        return IterableOrderedDict((k, convert_fn(v)) for k, v in data.items())
    return torch.utils.data._utils.collate.default_convert(data)
