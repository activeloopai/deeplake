import hub
import torch
import numpy as np
from hub.core.meta.encode.chunk_id import ChunkIdEncoder
from hub.util.iterable_ordered_dict import IterableOrderedDict


def raise_indra_installation_error(
    indra_installed: bool, indra_import_error: Exception
):
    if not indra_installed:
        raise ImportError(
            "This is an experimental feature that requires Hub deeplake package. To use it, you can run `pip install hub[deeplake]`."
        )
    if indra_import_error:
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
    primary_tensor = dataset[primary_tensor_name]
    chunk_id_encoder: ChunkIdEncoder = primary_tensor.chunk_engine.chunk_id_encoder
    enc_array = chunk_id_encoder.array
    num_chunks = chunk_id_encoder.num_chunks
    # pick chunks randomly, one by one
    chunk_order = np.random.choice(num_chunks, num_chunks, replace=False)
    schedule = []
    for chunk_idx in chunk_order:
        start_index = enc_array[chunk_idx - 1][1] + 1 if chunk_idx > 0 else 0
        last_index = enc_array[chunk_idx][1] + 1
        indexes = np.arange(start_index, last_index)
        schedule.extend(indexes)

    return schedule
