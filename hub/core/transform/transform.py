import hub
import math
from itertools import repeat
from typing import List

from hub.core.storage import MemoryProvider, LRUCache
from hub.core.compute import ThreadProvider, ProcessProvider, ComputeProvider

from hub.util.remove_cache import get_base_storage
from hub.util.dataset import try_flushing
from hub.util.transform import merge_chunk_id_encoders, merge_tensor_metas, store_shard

from hub.util.exceptions import (
    InvalidInputDataError,
    InvalidOutputDatasetError,
    MemoryDatasetNotSupportedError,
    UnsupportedSchedulerError,
)


def parallel(fn):
    def inner(*args, **kwargs):
        return TransformFunction(fn, args, kwargs)

    return inner


def compose(transform_functions):
    for fn in transform_functions:
        assert isinstance(fn, TransformFunction)
    return Pipeline(transform_functions)


class TransformFunction:
    def __init__(self, func, args, kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def eval(self, data_in, ds_out, workers, scheduler="threaded"):
        pipeline = Pipeline([self])
        pipeline.eval(data_in, ds_out, workers, scheduler)


class Pipeline:
    def __init__(self, transform_functions: List[TransformFunction]):
        self.transform_functions = transform_functions

    def __len__(self):
        return len(self.transform_functions)

    def eval(self, data_in, ds_out, workers, scheduler="threaded"):
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
            data_in_base_storage = get_base_storage(data_in.storage)
            cached_store = LRUCache(MemoryProvider(), data_in_base_storage, 0)
            data_in = hub.core.dataset.Dataset(
                storage=cached_store,
                index=data_in.index,
                read_only=data_in.read_only,
                log_loading=False,
            )
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

        self.run_pipeline(
            data_in,
            ds_out,
            tensors,
            compute,
            workers,
        )
        ds_out.storage.autoflush = initial_autoflush

    def run_pipeline(
        self,
        data_in,
        ds_out,
        tensors: List[str],
        compute: ComputeProvider,
        workers: int,
    ):
        """Runs the pipeline on the input data to produce output samples and stores in the dataset."""
        shard_size = math.ceil(len(data_in) / workers)
        shards = [
            data_in[i * shard_size : (i + 1) * shard_size] for i in range(workers)
        ]

        output_base_storage = get_base_storage(ds_out.storage)
        metas_and_encoders = compute.map(
            store_shard,
            zip(
                shards,
                repeat(output_base_storage),
                repeat(tensors),
                repeat(self),
            ),
        )

        all_workers_tensor_metas, all_workers_chunk_id_encoders = zip(
            *metas_and_encoders
        )
        merge_tensor_metas(all_workers_tensor_metas, ds_out)
        merge_chunk_id_encoders(all_workers_chunk_id_encoders, ds_out)


# @parallel
# def my_fn(a, b, c):
#     return a + b + c

# res = my_fn(1, 2, c=3).eval(None, None, None, None)
# import pickle
# pickle.dumps(res)
