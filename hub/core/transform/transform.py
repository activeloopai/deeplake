from hub.core.compute.provider import ComputeProvider
from hub.util.exceptions import (
    InvalidInputDataException,
    TensorMismatchException,
    UnsupportedSchedulerException,
)
from hub.core.compute import ThreadProvider, ProcessProvider
from hub.util.transform import merge_index_metas, merge_tensor_metas, transform_sample
from hub.core.meta.tensor_meta import TensorMeta
from hub.util.remove_cache import remove_all_cache
from hub.core.storage.lru_cache import LRUCache
from hub.core.storage.memory import MemoryProvider
from hub.core.meta.index_meta import IndexMeta
from hub.core.tensor import append_tensor
from hub.api.dataset import Dataset
from hub.constants import MB
from typing import Callable, Dict, List, Optional, Tuple
from itertools import repeat
import math


def transform(
    data_in,
    pipeline: List[Callable],
    ds_out: Dataset,
    pipeline_kwargs: Optional[List[Dict]] = None,
    scheduler: str = "threaded",
    workers: int = 1,
):

    """Initializes a new or existing dataset.

    Args:
        data_in: Input passed to the transform to generate output dataset. Should support __getitem__ and __len__. Can be a Hub dataset.
        pipeline (List[Callable]): A list of functions to apply to each element of data_in to generate output dataset.
            The output of each function should either be a dictionary or a list/tuple of dictionaries.
            The last function has added restriction that keys in the output of each sample should be same as the tensors present in the ds_out object.
        ds_out (Dataset): The dataset object to which the transform will get written.
            Should have all keys being generated in output already present as tensors. It's initial state should be either:-
            - Empty i.e. all tensors have no samples. In this case all samples are added to the dataset.
            - All tensors are populated and have sampe length. In this case new samples are appended to the dataset.
        pipeline_kwargs (List[dict], optional): A list containing extra arguments to be passed to the pipeline functions.
            If more kwargs than functions in pipeline, extra kwargs are ignored, if less kwargs, they are matched to only the starting functions.
            To use on non-continuous functions fill empty dict. Eg. pipeline=[fn1,fn2,fn3], kwargs=[{"a":5},{},{"c":1,"s":7}], only applies to fn1 and fn3.
        scheduler (str): The scheduler to be used to compute the transformation. Currently can be one of 'threaded' and 'processed'.
        workers (int): The number of workers to use for performing the transform. Defaults to 1.

    Raises:
        InvalidInputDataException: If ds_in passed to transform is invalid. It should support __getitem__ and __len__ operations.
        TensorMismatchException: If one or more of the outputs generated during transform contain different tensors than the ones present in the output 'ds_out' provided to transform.
        UnsupportedSchedulerException: If the scheduler passed is not recognized.
        InvalidTransformException: If the output of any step in a transformation isn't dictionary or a list/tuple of dictionaries.
    """
    ds_out.flush()

    if not hasattr(data_in, "__getitem__"):
        raise InvalidInputDataException("__getitem__")
    if not hasattr(data_in, "__len__"):
        raise InvalidInputDataException("__len__")

    # TODO: this check doesn't work currently. Will work once AL-1092 is merged and can be uncommented.
    # for tensor in tensors:
    #     assert len(ds_out[tensor]) == len(ds_out)

    workers = max(workers, 1)

    if scheduler == "threaded":
        compute: ComputeProvider = ThreadProvider(workers)
    elif scheduler == "processed":
        compute = ProcessProvider(workers)
    else:
        raise UnsupportedSchedulerException(scheduler)

    base_storage = remove_all_cache(ds_out.storage)
    tensors = set(ds_out.meta.tensors)

    pipeline_kwargs = pipeline_kwargs or []
    pipeline_kwargs = pipeline_kwargs[0 : len(pipeline)]
    pipeline_kwargs += [{}] * (len(pipeline) - len(pipeline_kwargs))

    store(data_in, base_storage, tensors, compute, workers, pipeline, pipeline_kwargs)
    ds_out.clear_cache()
    ds_out._load_meta()


def store_shard(transform_input: Tuple):
    """Takes a shard of the original data and iterates through it, producing chunks."""
    data_shard, size, storage, tensors, pipeline, pipeline_kwargs = transform_input

    # storing the metas in memory to merge later
    all_index_meta = {key: IndexMeta.create(key, MemoryProvider()) for key in tensors}
    all_tensor_meta = {key: TensorMeta.create(key, MemoryProvider()) for key in tensors}

    # separate cache for each tensor to prevent frequent flushing, 32 MB ensures only full chunks are written.
    storage_map = {key: LRUCache(MemoryProvider(), storage, 32 * MB) for key in tensors}

    # will be simply range(len(data_shard)) after AL 1092
    for i in range(min(len(data_shard), size)):
        sample = data_shard[i]
        if isinstance(sample, Dataset):
            sample_dict = {}
            for k in sample.tensors:
                sample_dict[k] = sample[k].numpy()
            sample = sample_dict

        # always a list of dicts
        results = transform_sample(sample, pipeline, pipeline_kwargs)
        for result in results:
            if set(result.keys()) != tensors:
                raise TensorMismatchException(list(tensors), list(result.keys()))
            for key, value in result.items():
                append_tensor(
                    value,
                    key,
                    storage_map[key],
                    index_meta=all_index_meta[key],
                    tensor_meta=all_tensor_meta[key],
                )

    for tensor in tensors:
        storage_map[tensor].flush()

    return all_index_meta, all_tensor_meta


def store(data_in, storage, tensors, compute, workers, pipeline, pipeline_kwargs):
    shard_size = math.ceil(len(data_in) / workers)
    shards = [data_in[i * shard_size : (i + 1) * shard_size] for i in range(workers)]

    # TODO: hacky way to get around improper length of hub dataset slices, can be removed once AL-1092 gets done
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
