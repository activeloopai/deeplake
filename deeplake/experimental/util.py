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
            "This is an experimental feature that requires libdeeplake package. libdeeplake is available only on linux for python versions 3.6 through 3.10 and on macos for python versions 3.7 through 3.10"
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
        index_set = set(range(start, stop, step))
    elif isinstance(slice_, (list, tuple)):
        index_set = set(slice_)
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

    schedule = [idx for idx in schedule if idx in index_set]
    return schedule


def remove_tiled_samples(dataset, slice_):
    found_tiled_samples = False
    for tensor in dataset.tensors.values():
        chunk_engine: ChunkEngine = tensor.chunk_engine
        if chunk_engine.tile_encoder_exists:
            tiles = set(chunk_engine.tile_encoder.entries.keys())
            if len(tiles) > 0:
                found_tiled_samples = True
                if isinstance(slice_, slice):
                    start = slice_.start if slice_.start is not None else 0
                    stop = (
                        slice_.stop if slice_.stop is not None else tensor.num_samples
                    )
                    step = slice_.step if slice_.step is not None else 1
                    slice_ = list(range(start, stop, step))
                if isinstance(slice_, (list, tuple)):
                    slice_ = [idx for idx in slice_ if idx not in tiles]

    if found_tiled_samples:
        warnings.warn(
            "One or more tiled samples (big samples that span across multiple chunks) were found in the dataset. These samples are currently not supported for query and dataloader and will be ignored."
        )

    return slice_


def verify_base_storage(dataset):
    if isinstance(dataset.base_storage, (GCSProvider, GDriveProvider, MemoryProvider)):
        raise ValueError(
            "GCS, Google Drive and Memory datasets are not supported for experimental features currently."
        )
