from hub.core.compute import ThreadProvider, ProcessProvider
from hub.util.transform import merge_index_metas, merge_tensor_metas, transform_sample
from hub.core.meta.tensor_meta import TensorMeta
from hub.util.remove_cache import remove_all_cache
from hub.core.storage.lru_cache import LRUCache
from hub.core.storage.memory import MemoryProvider
from hub.core.meta.index_meta import IndexMeta
from hub.core.tensor import add_samples_to_tensor
from hub.api.dataset import Dataset
from hub.constants import MB
from typing import Callable, Dict, List
from itertools import repeat
import math

# TODO Ensure that all outputs have the same schema


def transform(
    data_in,
    pipeline: List[Callable],
    ds_out,
    pipeline_kwargs: List[Dict] = None,
    scheduler: str = "threaded",
    workers: int = 1,
):
    ds_out.flush()

    assert hasattr(data_in, "__getitem__")  # TODO better check
    assert hasattr(data_in, "__len__")  # TODO better check

    workers = max(workers, 1)

    if scheduler == "threaded":
        compute = ThreadProvider(workers)
    elif scheduler == "processed":
        compute = ProcessProvider(workers)
    else:
        raise Exception  # TODO better exception

    base_storage = remove_all_cache(ds_out.storage)
    tensors = set(ds_out.meta.tensors)

    # this check doesn't work currently. Will work once AL-1092 is merged and can be uncommented
    # for tensor in tensors:
    #     assert len(ds_out[tensor]) == len(ds_out)

    # TODO handle kwargs properly
    pipeline_kwargs = [{} for i in range(len(pipeline))]
    store(data_in, base_storage, tensors, compute, workers, pipeline, pipeline_kwargs)


def store_shard(inp):
    """Takes a shard of the original data and iterates through it, producing chunks."""
    data_shard, size, storage, tensors, pipeline, pipeline_kwargs = inp

    # storing the metas in memory to merge later
    all_index_meta = {key: IndexMeta.create(key, MemoryProvider()) for key in tensors}
    all_tensor_meta = {key: TensorMeta.create(key, MemoryProvider()) for key in tensors}

    # separate cache for each tensor to prevent frequent flushing, 32 MB ensures only full chunks are written.
    storage_map = {key: LRUCache(MemoryProvider(), storage, 32 * MB) for key in tensors}

    # will be simply range(len(data_shard)) after AL 1092
    for i in range(min(len(data_shard), size)):
        sample = data_shard[i]
        if isinstance(sample, Dataset):
            sample = sample.numpy()

        results = transform_sample(
            sample, pipeline, pipeline_kwargs, tensors
        )  # always a list of dicts
        for result in results:
            for key, value in result.items():
                add_samples_to_tensor(
                    value,
                    key,
                    storage_map[key],
                    batched=False,
                    index_meta=all_index_meta[key],
                    tensor_meta=all_tensor_meta[key],
                )

    for tensor in tensors:
        storage_map[tensor].flush()

    return all_index_meta, all_tensor_meta


def store(data_in, storage, tensors, compute, workers, pipeline, pipeline_kwargs):
    shard_size = math.ceil(len(data_in) / workers)
    shards = [data_in[i * shard_size : (i + 1) * shard_size] for i in range(workers)]

    # hacky way to get around improper length of hub dataset slices, can be removed once AL-1092 gets done
    size_list = [shard_size for _ in range(workers)]
    extra = shard_size * workers - len(data_in)
    if size_list:
        size_list[-1] -= extra

    all_workers_metas = compute.map(
        store_shard,
        zip(
            shards,
            size_list,
            repeat(storage),
            repeat(tensors),
            repeat(pipeline),
            repeat(pipeline_kwargs),
        ),
    )
    all_workers_metas = list(all_workers_metas)
    all_workers_index_meta, all_workers_tensor_meta = zip(*all_workers_metas)

    merge_tensor_metas(all_workers_tensor_meta, storage, tensors)
    merge_index_metas(all_workers_index_meta, storage, tensors)
