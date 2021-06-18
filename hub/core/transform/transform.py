from hub.util.exceptions import (
    InvalidInputDataError,
    TensorMismatchError,
    UnsupportedSchedulerError,
)
from hub.util.transform import (
    pad_or_shrink_kwargs,
    load_updated_meta,
    merge_index_metas,
    merge_tensor_metas,
    transform_sample,
)
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple
from hub.util.remove_cache import get_base_storage
from hub.core.storage import LRUCache, MemoryProvider, StorageProvider
from hub.core.compute import ThreadProvider, ProcessProvider, ComputeProvider
from hub.core.meta.tensor_meta import TensorMeta
from hub.core.meta.index_meta import IndexMeta
from hub.core.tensor import append_tensor
from hub.constants import MB
from itertools import repeat
from hub import Dataset
import math


def transform(
    data_in,
    pipeline: Sequence[Callable],
    ds_out: Dataset,
    pipeline_kwargs: Optional[Sequence[Dict]] = None,
    scheduler: str = "threaded",
    workers: int = 1,
):

    """Transforms the data_in to produce an output dataset ds_out using one or more workers.
    Useful for generating new datasets or converting datasets from one format to another efficiently.

    Eg.
    transform(["xyz.png", "abc.png"], [load_img, mirror], ds, workers=5) # loads images, mirrors them and stores them in ds which is a Hub dataset.
    transform(["xyz.png", "abc.png"], [load_img, rotate], ds, [{"grayscale":True}, {"angle":30}]) # applies grayscale arg to load_img and angle to rotate


    Args:
        data_in: Input passed to the transform to generate output dataset. Should support __getitem__ and __len__. Can be a Hub dataset.
        pipeline (Sequence[Callable]): A Sequence of functions to apply to each element of data_in to generate output dataset.
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
        InvalidInputDataError: If ds_in passed to transform is invalid. It should support __getitem__ and __len__ operations.
        TensorMismatchError: If one or more of the outputs generated during transform contain different tensors than the ones present in the output 'ds_out' provided to transform.
        UnsupportedSchedulerError: If the scheduler passed is not recognized.
        InvalidTransformOutputError: If the output of any step in a transformation isn't dictionary or a list/tuple of dictionaries.
    """
    ds_out.flush()

    if not hasattr(data_in, "__getitem__"):
        raise InvalidInputDataError("__getitem__")
    if not hasattr(data_in, "__len__"):
        raise InvalidInputDataError("__len__")

    # TODO: this check doesn't work currently. Will work once AL-1092 is merged and can be uncommented.
    # for tensor in tensors:
    #     assert len(ds_out[tensor]) == len(ds_out)

    workers = max(workers, 1)

    if scheduler == "threaded":
        compute: ComputeProvider = ThreadProvider(workers)
    elif scheduler == "processed":
        compute = ProcessProvider(workers)
    else:
        raise UnsupportedSchedulerError(scheduler)

    base_storage = get_base_storage(ds_out.storage)
    tensors = set(ds_out.meta.tensors)

    pipeline_kwargs = pad_or_shrink_kwargs(pipeline_kwargs, pipeline)

    run_pipeline(
        data_in, base_storage, tensors, compute, workers, pipeline, pipeline_kwargs
    )
    load_updated_meta(ds_out)


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

        results = transform_sample(sample, pipeline, pipeline_kwargs)
        for result in results:
            if set(result.keys()) != tensors:
                raise TensorMismatchError(list(tensors), list(result.keys()))
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
        all_tensor_meta[tensor] = all_tensor_meta[tensor].to_dict()
        all_index_meta[tensor] = all_index_meta[tensor].to_dict()

    return all_index_meta, all_tensor_meta


def run_pipeline(
    data_in,
    storage: StorageProvider,
    tensors: Set[str],
    compute: ComputeProvider,
    workers: int,
    pipeline: Sequence[Callable],
    pipeline_kwargs: List[Dict],
):
    """Runs the pipeline on the input data to produce output samples and stores in the dataset."""
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
