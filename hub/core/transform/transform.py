from hub.core.compute.provider import ComputeProvider
import hub
import math
from itertools import repeat
from typing import List

from hub.core.storage import MemoryProvider, LRUCache
from hub.util.remove_cache import get_base_storage, get_dataset_with_zero_size_cache
from hub.util.dataset import try_flushing
from hub.util.transform import (
    check_transform_data_in,
    check_transform_ds_out,
    merge_all_chunk_id_encoders,
    merge_all_tensor_metas,
    store_data_slice,
)
from hub.util.exceptions import (
    InvalidInputDataError,
    InvalidOutputDatasetError,
    MemoryDatasetNotSupportedError,
)
from hub.util.compute import get_compute_provider


class TransformFunction:
    def __init__(self, func, args, kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def eval(self, data_in, ds_out, num_workers=1, scheduler="threaded"):
        pipeline = Pipeline([self])
        pipeline.eval(data_in, ds_out, num_workers, scheduler)


class Pipeline:
    def __init__(self, transform_functions: List[TransformFunction]):
        if not transform_functions:
            raise Exception  # TODO: Proper exception
        self.transform_functions = transform_functions

    def __len__(self):
        return len(self.transform_functions)

    def eval(self, data_in, ds_out, num_workers: int = 1, scheduler="threaded"):
        """Evaluates the pipeline on data_in to produce an output dataset ds_out.

        Args:
            data_in: Input passed to the transform to generate output dataset. Should support __getitem__ and __len__. Can be a Hub dataset.
            ds_out (Dataset): The dataset object to which the transform will get written.
                Should have all keys being generated in output already present as tensors. It's initial state should be either:-
                - Empty i.e. all tensors have no samples. In this case all samples are added to the dataset.
                - All tensors are populated and have sampe length. In this case new samples are appended to the dataset.
            num_workers (int): The number of workers to use for performing the transform. Defaults to 1.
            scheduler (str): The scheduler to be used to compute the transformation. Supported values include: 'threaded' and 'processed'.

        Raises:
            InvalidInputDataError: If data_in passed to transform is invalid. It should support __getitem__ and __len__ operations.
            InvalidOutputDatasetError: If all the tensors of ds_out passed to transform don't have the same length.
            TensorMismatchError: If one or more of the outputs generated during transform contain different tensors than the ones present in 'ds_out' provided to transform.
            UnsupportedSchedulerError: If the scheduler passed is not recognized. Supported values include: 'threaded' and 'processed'.
            InvalidTransformOutputError: If the output of any step in a transformation isn't dictionary or a list/tuple of dictionaries.
            MemoryDatasetNotSupportedError: If ds_out is a Hub dataset with memory as underlying storage and the scheduler is not threaded.
        """
        if isinstance(data_in, hub.core.dataset.Dataset):
            data_in = get_dataset_with_zero_size_cache(data_in)

        check_transform_data_in(data_in)

        check_transform_ds_out(ds_out)
        ds_out.flush()
        initial_autoflush = ds_out.storage.autoflush
        ds_out.storage.autoflush = False

        output_base_storage = get_base_storage(ds_out.storage)
        if isinstance(output_base_storage, MemoryProvider) and scheduler != "threaded":
            # TODO: do this for input data too
            raise MemoryDatasetNotSupportedError(scheduler)

        tensors = list(ds_out.tensors)
        compute_provider = get_compute_provider(scheduler, num_workers)

        self.run_pipeline(data_in, ds_out, tensors, compute_provider, num_workers)
        ds_out.storage.autoflush = initial_autoflush

    def run_pipeline(
        self,
        data_in,
        ds_out,
        tensors: List[str],
        compute: ComputeProvider,
        num_workers: int,
    ):
        """Runs the pipeline on the input data to produce output samples and stores in the dataset."""
        size = math.ceil(len(data_in) / num_workers)
        slices = [data_in[i * size : (i + 1) * size] for i in range(num_workers)]

        output_base_storage = get_base_storage(ds_out.storage)
        metas_and_encoders = compute.map(
            store_data_slice,
            zip(slices, repeat(output_base_storage), repeat(tensors), repeat(self)),
        )

        all_tensor_metas, all_chunk_id_encoders = zip(*metas_and_encoders)
        merge_all_tensor_metas(all_tensor_metas, ds_out)
        merge_all_chunk_id_encoders(all_chunk_id_encoders, ds_out)


def compose(transform_functions: List[TransformFunction]):
    for fn in transform_functions:
        assert isinstance(fn, TransformFunction)  # TODO add exception
    return Pipeline(transform_functions)


def compute(fn):
    def inner(*args, **kwargs):
        return TransformFunction(fn, args, kwargs)

    return inner
