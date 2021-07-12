from hub.util.iterable_ordered_dict import IterableOrderedDict


def _collate_fn(batch):
    import torch

    elem = batch[0]
    if isinstance(elem, IterableOrderedDict):
        return IterableOrderedDict(
            (key, _collate_fn([d[key] for d in batch])) for key in elem.keys()
        )
    return torch.utils.data._utils.collate.default_collate(batch)


def _convert_fn(data):
    import torch

    elem_type = type(data)
    if isinstance(data, IterableOrderedDict):
        return IterableOrderedDict((k, _convert_fn(v)) for k, v in data.items())
    return torch.utils.data._utils.collate.default_convert(data)
