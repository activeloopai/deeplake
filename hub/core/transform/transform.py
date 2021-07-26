from hub.util.cache_chain import generate_chain
from hub.util.keys import get_tensor_meta_key
from numpy import dtype
from hub.util.dataset import try_flushing
from hub.util.exceptions import (
    InvalidInputDataError,
    InvalidOutputDatasetError,
    MemoryDatasetNotSupportedError,
    TensorMismatchError,
    UnsupportedSchedulerError,
)
from hub.util.transform import (
    merge_all_chunk_engines,
    pipeline_to_list,
    transform_sample,
)
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple, Union
from hub.util.remove_cache import get_base_storage
from hub.core.storage import LRUCache, MemoryProvider, StorageProvider
from hub.core.compute import ThreadProvider, ProcessProvider, ComputeProvider
from hub.core.chunk_engine import ChunkEngine
from hub.core.meta.tensor_meta import TensorMeta
from hub.constants import MB
from itertools import repeat
import hub
import math


def transform(
    data_in,
    pipeline: Union[Callable, Sequence[Callable]],
    ds_out: hub.core.dataset.Dataset,
    pipeline_kwargs: Optional[Union[Dict, Sequence[Dict]]] = None,
    scheduler: str = "threaded",
    workers: int = 1,
):

    """Transforms the data_in to produce an output dataset ds_out using one or more workers.
    Useful for generating new datasets or converting datasets from one format to another efficiently.

    Examples:
        >>> transform(["xyz.png", "abc.png"], load_img, ds, workers=2, scheduler="processed")  # single function in pipleline
        >>> transform(["xyz.png", "abc.png"], [load_img, mirror], ds, workers=5)  # loads images, mirrors them and stores them in ds which is a Hub dataset.
        >>> transform(["xyz.png", "abc.png"], [load_img, rotate], ds, [{"grayscale":True}, {"angle":30}])  # applies grayscale arg to load_img and angle to rotate

    Args:
        data_in: Input passed to the transform to generate output dataset. Should support __getitem__ and __len__. Can be a Hub dataset.
        pipeline (Union[Callable, Sequence[Callable]]): A single function or a sequence of functions to apply to each element of data_in to generate output dataset.
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
    if isinstance(data_in, hub.core.dataset.Dataset):
        try_flushing(data_in)
        # data_in_base_storage = get_base_storage(data_in.storage)
        # data_in = Dataset(
        #     storage=data_in_base_storage, memory_cache_size=0, local_cache_size=0
        # )
    if ds_out._read_only:
        raise InvalidOutputDatasetError
    ds_out.flush()
    initial_autoflush = ds_out.storage.autoflush
    ds_out.storage.autoflush = False
    if not hasattr(data_in, "__getitem__"):
        raise InvalidInputDataError("__getitem__")
    if not hasattr(data_in, "__len__"):
        raise InvalidInputDataError("__len__")

    tensors = list(ds_out.meta.tensors)
    for tensor in tensors:
        if len(ds_out[tensor]) != len(ds_out):
            raise InvalidOutputDatasetError(
                "One or more tensors of the ds_out have different lengths. Transform only supports ds_out having same number of samples for each tensor (This includes empty datasets that have 0 samples per tensor)."
            )

    workers = max(workers, 1)

    if scheduler == "threaded":
        compute: ComputeProvider = ThreadProvider(workers)
    elif scheduler == "processed":
        compute = ProcessProvider(workers)
    else:
        raise UnsupportedSchedulerError(scheduler)

    output_base_storage = get_base_storage(ds_out.storage)
    if isinstance(output_base_storage, MemoryProvider) and scheduler != "threaded":
        raise MemoryDatasetNotSupportedError(scheduler)

    pipeline, pipeline_kwargs = pipeline_to_list(pipeline, pipeline_kwargs)

    run_pipeline(
        data_in,
        ds_out,
        tensors,
        compute,
        workers,
        pipeline,
        pipeline_kwargs,
    )
    # load_updated_meta(ds_out)
    ds_out.storage.autoflush = initial_autoflush


def store_shard(transform_input: Tuple):
    """Takes a shard of the original data and iterates through it, producing chunks."""
    (
        data_shard,
        chunk_engines,
        pipeline,
        pipeline_kwargs,
    ) = transform_input
    tensors = set(chunk_engines.keys())

    # storing the metas in memory to merge later
    all_chunk_engines = {}
    all_caches = {}
    for tensor in tensors:
        memory_cache = LRUCache(MemoryProvider(), MemoryProvider(), 32 * MB)
        chunk_engine = chunk_engines[tensor]
        existing_meta = chunk_engine.tensor_meta
        new_tensor_meta = TensorMeta(
            htype=existing_meta.htype,
            dtype=existing_meta.dtype,
            sample_compression=existing_meta.sample_compression,
        )
        meta_key = get_tensor_meta_key(tensor)
        memory_cache[meta_key] = new_tensor_meta  # type: ignore
        actual_storage = get_base_storage(chunk_engine.cache)
        new_cache = LRUCache(MemoryProvider(), actual_storage, 32 * MB)
        new_cache.autoflush = False
        chunk_engine = ChunkEngine(
            tensor, new_cache, chunk_engine.max_chunk_size, memory_cache
        )
        all_chunk_engines[tensor] = chunk_engine
        all_caches[tensor] = new_cache

    # TODO separate cache for data_shard if it's a Dataset

    for i in range(len(data_shard)):
        sample = data_shard[i]
        if isinstance(sample, hub.core.dataset.Dataset):
            sample = {key: sample[key].numpy() for key in sample.tensors}

        results = transform_sample(sample, pipeline, pipeline_kwargs)
        for result in results:
            if set(result.keys()) != tensors:
                raise TensorMismatchError(list(tensors), list(result.keys()))
            for key, value in result.items():
                all_chunk_engines[key].append(value)

    all_tensor_metas = {}
    all_chunk_id_encoders = {}
    for tensor in tensors:
        all_caches[tensor].flush()
    #     all_tensor_metas[tensor] = all_chunk_engines[tensor].tensor_meta
    #     all_chunk_id_encoders[tensor] = all_chunk_engines[tensor].chunk_id_encoder

    return all_chunk_engines


def run_pipeline(
    data_in,
    ds_out,
    tensors: List[str],
    compute: ComputeProvider,
    workers: int,
    pipeline: Sequence[Callable],
    pipeline_kwargs: List[Dict],
):
    """Runs the pipeline on the input data to produce output samples and stores in the dataset."""
    output_base_storage = get_base_storage(ds_out.storage)

    shard_size = math.ceil(len(data_in) / workers)
    shards = [data_in[i * shard_size : (i + 1) * shard_size] for i in range(workers)]
    print("sharding complete")
    init_chunk_engines = {
        tensor: ChunkEngine(
            tensor, LRUCache(MemoryProvider(), output_base_storage, 16 * MB)
        )
        for tensor in tensors
    }
    all_workers_chunk_engines = compute.map(
        store_shard,
        zip(
            shards,
            repeat(init_chunk_engines),
            repeat(pipeline),
            repeat(pipeline_kwargs),
        ),
    )
    merge_all_chunk_engines(all_workers_chunk_engines, ds_out)
