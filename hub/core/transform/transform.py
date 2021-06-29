from hub.util.exceptions import (
    InvalidInputDataError,
    InvalidOutputDatasetError,
    MemoryDatasetNotSupportedError,
    TensorMismatchError,
    UnsupportedSchedulerError,
)
from hub.util.transform import (
    pipeline_to_list,
    load_updated_meta,
    merge_index_metas,
    merge_tensor_metas,
    transform_sample,
)
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple, Union
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
    pipeline: Union[Callable, Sequence[Callable]],
    ds_out: Dataset,
    pipeline_kwargs: Optional[Union[Dict, Sequence[Dict]]] = None,
    scheduler: str = "threaded",
    workers: int = 1,
):

    """Transforms the data_in to produce an output dataset ds_out using one or more workers.
    Useful for generating new datasets or converting datasets from one format to another efficiently.
    Eg.
    transform(["xyz.png", "abc.png"], load_img, ds, workers=2, scheduler="processed") # single function in pipleline
    transform(["xyz.png", "abc.png"], [load_img, mirror], ds, workers=5) # loads images, mirrors them and stores them in ds which is a Hub dataset.
    transform(["xyz.png", "abc.png"], [load_img, rotate], ds, [{"grayscale":True}, {"angle":30}]) # applies grayscale arg to load_img and angle to rotate
    Args:
        data_in: Input passed to the transform to generate output dataset. Should support __getitem__ and __len__. Can be a Hub dataset.
        pipeline (Union[Callabel, Sequence[Callable]]): A single function or a sequence of functions to apply to each element of data_in to generate output dataset.
            The output of each function should either be a dictionary or a list/tuple of dictionaries.
            The last function has added restriction that keys in the output of each sample should be same as the tensors present in the ds_out object.
        ds_out (Dataset): The dataset object to which the transform will get written.
            Should have all keys being generated in output already present as tensors. It's initial state should be either:-
            - Empty i.e. all tensors have no samples. In this case all samples are added to the dataset.
            - All tensors are populated and have sampe length. In this case new samples are appended to the dataset.
        pipeline_kwargs (Union[Dict, Sequence[Dict]], optional): A single dictionary or a sequence of dictionaries containing extra arguments to be passed to the pipeline functions.
            If more kwargs than functions in pipeline, extra kwargs are ignored, if less kwargs, they are matched to only the starting functions.
            To use on non-continuous functions fill empty dict. Eg. pipeline=[fn1,fn2,fn3], kwargs=[{"a":5},{},{"c":1,"s":7}], only applies to fn1 and fn3.
        scheduler (str): The scheduler to be used to compute the transformation. Currently can be one of 'threaded' and 'processed'.
        workers (int): The number of workers to use for performing the transform. Defaults to 1.
    Raises:
        InvalidInputDataError: If data_in passed to transform is invalid. It should support __getitem__ and __len__ operations.
        InvalidOutputDatasetError: If all the tensors of ds_out passed to transform don't have the same length.
        TensorMismatchError: If one or more of the outputs generated during transform contain different tensors than the ones present in 'ds_out' provided to transform.
        UnsupportedSchedulerError: If the scheduler passed is not recognized.
        InvalidTransformOutputError: If the output of any step in a transformation isn't dictionary or a list/tuple of dictionaries.
        MemoryDatasetNotSupportedError: If ds_out is a Hub dataset with memory as underlying storage and the scheduler is not threaded.
    """
    if isinstance(data_in, Dataset):
        data_in.flush()
        data_in_base_storage = get_base_storage(data_in.storage)
        data_in = Dataset(
            storage=data_in_base_storage, memory_cache_size=0, local_cache_size=0
        )
    ds_out.flush()

    if not hasattr(data_in, "__getitem__"):
        raise InvalidInputDataError("__getitem__")
    if not hasattr(data_in, "__len__"):
        raise InvalidInputDataError("__len__")

    tensors = set(ds_out.meta.tensors)
    for tensor in tensors:
        if len(ds_out[tensor]) != len(ds_out):
            raise InvalidOutputDatasetError

    workers = max(workers, 1)

    if scheduler == "threaded":
        compute: ComputeProvider = ThreadProvider(workers)
    elif scheduler == "processed":
        compute = ProcessProvider(workers)
    else:
        raise UnsupportedSchedulerError(scheduler)

    base_storage = get_base_storage(ds_out.storage)
    if isinstance(base_storage, MemoryProvider) and scheduler != "threaded":
        raise MemoryDatasetNotSupportedError(scheduler)

    pipeline, pipeline_kwargs = pipeline_to_list(pipeline, pipeline_kwargs)

    run_pipeline(
        data_in, base_storage, tensors, compute, workers, pipeline, pipeline_kwargs
    )
    load_updated_meta(ds_out)


def store_shard(transform_input: Tuple):
    """Takes a shard of the original data and iterates through it, producing chunks."""
    data_shard, storage, tensor_metas, pipeline, pipeline_kwargs = transform_input

    tensors = tensor_metas.keys()
    # storing the metas in memory to merge later
    all_index_meta = {key: IndexMeta.create(key, MemoryProvider()) for key in tensors}
    all_tensor_meta = {}
    for tensor in tensors:
        all_tensor_meta[tensor] = TensorMeta.create(tensor, MemoryProvider())
        all_tensor_meta[tensor].htype = tensor_metas[tensor].htype
        all_tensor_meta[tensor].dtype = tensor_metas[tensor].dtype
        all_tensor_meta[tensor].sample_compression = tensor_metas[
            tensor
        ].sample_compression

    # separate cache for each tensor to prevent frequent flushing, 32 MB ensures only full chunks are written.
    storage_map = {key: LRUCache(MemoryProvider(), storage, 32 * MB) for key in tensors}

    # TODO separate cache for data_in if it's a Dataset

    for i in range(len(data_shard)):
        sample = data_shard[i]
        if isinstance(sample, Dataset):
            sample = {key: sample[key].numpy() for key in sample.tensors}

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

    init_tensor_metas = {tensor: TensorMeta.load(tensor, storage) for tensor in tensors}
    all_workers_metas = compute.map(
        store_shard,
        zip(
            shards,
            repeat(storage),
            repeat(init_tensor_metas),
            repeat(pipeline),
            repeat(pipeline_kwargs),
        ),
    )
    all_workers_metas = list(all_workers_metas)
    all_workers_index_meta, all_workers_tensor_meta = zip(*all_workers_metas)

    merge_tensor_metas(all_workers_tensor_meta, storage, tensors)
    merge_index_metas(all_workers_index_meta, storage, tensors)
