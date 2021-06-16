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

# TODO Ensure that all outputs have the same schema


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
        data_in: The input iterable passed to the transform to generate output dataset. Can be a Hub dataset.
            Should support slicing using __getitem__ and __len__.
        pipeline (List[Callable]): A list of functions to apply to each element of data_in to generate output dataset.
            The output of each function (including the last one) should either be a dictionary or a list/tuple of dictionaries.
            The output of last function has the added restriction that the keys in the output dictionaries of each sample should
            be exactly the same as the tensors present in the ds_out object.
        ds_out (Dataset): The dataset object used to which the transformation will get written.
            This should have all the keys being generated in the output already present before being passed.
            The initial state of the dataset can be:-
            1. Empty i.e. all tensors are created but there are no samples. In this case all samples are simply added to the dataset.
            After transform is complete, ds_out has x samples in each tensor, where x is the number of outputs generated on applying pipeline on data_in.
            2. All tensors are populated, but all have sampe length, say n. In this case all samples are simply appended to the dataset.
            After transform is complete, ds_out has n + x samples in each tensor, where x is the number of outputs generated on applying pipeline on data_in.
            3. All tensors are populated, but don't have the same length. This case is NOT supported.
        pipeline_kwargs (List[dict], optional): A list containing extra arguments to be passed to the pipeline functions.
            If this is None no extra arguments are passed to the pipeline functions.
            If this contains exactly the same number of kwargs as number of functions in pipeline, corresponding kwargs are applied to each function.
            If this contains less kwargs than the number of functions, only the starting functions upto the length of pipeline_kwargs get kwargs.
            If this contains more kwargs than the number of functions, the extra kwargs are ignore.
            Tip: If you want to apply kwargs to non-continuos functions, then use empty dictionaries in the middle.
            For example:- transform(data_in, [fn1, fn2, fn3], ds_out, [{"a":5, "b":6}, {}, {"c":11, "s":7}]), only applies kwargs to fn1 and fn3.
        scheduler (str, optional): The scheduler to be used to compute the transformation.
            Currently can be one of:-
            threaded: Uses multithreading to perform the transform. Best applicable for I/O intensive transforms.
            processed: Uses multiprocessing to perform the transform. Best applicable for CPU intensive transforms. Currently doesn't work with S3 or Hub cloud datasets.
        workers (int, optional): The number of workers to use for performing the transform. Defaults to 1.

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
        compute = ThreadProvider(workers)
    elif scheduler == "processed":
        compute = ProcessProvider(workers)
    else:
        raise UnsupportedSchedulerException(scheduler)

    base_storage = remove_all_cache(ds_out.storage)
    tensors = set(ds_out.meta.tensors)

    # TODO handle kwargs properly
    pipeline_kwargs = [{} for _ in range(len(pipeline))]
    store(data_in, base_storage, tensors, compute, workers, pipeline, pipeline_kwargs)


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
            sample = sample.numpy()

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
