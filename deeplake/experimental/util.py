import deeplake
import torch
from typing import Optional
import numpy as np
from deeplake.util.iterable_ordered_dict import IterableOrderedDict


def raise_indra_installation_error(indra_import_error: Optional[Exception] = None):
    if not indra_import_error:
        raise ImportError(
            "This is an experimental feature that requires Hub deeplake package. To use it, you can run `pip install hub[deeplake]`."
        )
    raise ImportError(
        "Error while importing C++ backend. One of the dependencies might not be installed."
    ) from indra_import_error


def collate_fn(batch):
    elem = batch[0]

    if isinstance(elem, IterableOrderedDict):
        return IterableOrderedDict(
            (key, collate_fn([d[key] for d in batch])) for key in elem.keys()
        )
    if isinstance(elem, np.ndarray) and elem.dtype.type is np.str_:
        batch = [it.item() for it in batch]

    return torch.utils.data._utils.collate.default_collate(batch)
