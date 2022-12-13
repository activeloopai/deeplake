from collections import defaultdict
import torch
from typing import Optional
import numpy as np
from deeplake.core.meta.encode.chunk_id import ChunkIdEncoder
from deeplake.util.iterable_ordered_dict import IterableOrderedDict
from deeplake.core.chunk_engine import ChunkEngine
from deeplake.core.storage import GCSProvider, GDriveProvider, MemoryProvider
import warnings


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


def find_primary_tensor(dataset):
    current_max_size = 0
    primary_tensor_name = None
    for tensor_key, tensor in dataset.tensors.items():
        max_shape = tensor.meta.max_shape
        max_size = np.prod(max_shape)
        if max_size > current_max_size:
            current_max_size = max_size
            primary_tensor_name = tensor_key

    return primary_tensor_name


def create_fetching_schedule(dataset, primary_tensor_name):
    slice_ = dataset.index.values[0].value
    if isinstance(slice_, int):
        return None
    elif isinstance(slice_, slice):
        start = slice_.start if slice_.start is not None else 0
        stop = slice_.stop if slice_.stop is not None else dataset.min_len
        step = slice_.step if slice_.step is not None else 1
        index_struct = set(range(start, stop, step))
    elif isinstance(slice_, (list, tuple)):
        index_struct = defaultdict(lambda: 0)
        for item in slice_:
            index_struct[item] += 1
    primary_tensor = dataset[primary_tensor_name]
    chunk_id_encoder: ChunkIdEncoder = primary_tensor.chunk_engine.chunk_id_encoder
    enc_array = chunk_id_encoder.array
    num_chunks = chunk_id_encoder.num_chunks
    # pick chunks randomly, one by one
    chunk_order = np.random.choice(num_chunks, num_chunks, replace=False)
    schedule = []
    for chunk_idx in chunk_order:
        start_index = int(enc_array[chunk_idx - 1][1]) + 1 if chunk_idx > 0 else 0
        last_index = int(enc_array[chunk_idx][1]) + 1
        indexes = np.arange(start_index, last_index)
        schedule.extend(indexes)

    if isinstance(index_struct, set):
        schedule = [idx for idx in schedule if idx in index_struct]
    elif isinstance(index_struct, dict):
        nested_schedule = [
            [idx] * index_struct[idx] for idx in schedule if idx in index_struct
        ]
        schedule = []
        for indexes_list in nested_schedule:
            schedule.extend(indexes_list)
    return schedule


def verify_base_storage(dataset):
    if isinstance(dataset.base_storage, (GCSProvider, GDriveProvider, MemoryProvider)):
        raise ValueError(
            "GCS, Google Drive and Memory datasets are not supported for experimental features currently."
        )
