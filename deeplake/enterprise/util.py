import torch
from typing import Optional
import numpy as np
from deeplake.util.iterable_ordered_dict import IterableOrderedDict
from deeplake.core.storage import GCSProvider, GDriveProvider, MemoryProvider


def raise_indra_installation_error(indra_import_error: Optional[Exception] = None):
    if not indra_import_error:
        raise ImportError(
            "This is an enterprise feature that requires libdeeplake package which can be installed using pip install deeplake[enterprise]. libdeeplake is available only on linux for python versions 3.6 through 3.10 and on macos for python versions 3.7 through 3.10"
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


def verify_base_storage(dataset):
    if isinstance(dataset.base_storage, (GCSProvider, GDriveProvider, MemoryProvider)):
        raise ValueError(
            "GCS, Google Drive and Memory datasets are not supported for experimental features currently."
        )
