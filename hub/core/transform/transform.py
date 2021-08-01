import hub
import math
from typing import List
from itertools import repeat
from hub.core.compute.provider import ComputeProvider
from hub.util.compute import get_compute_provider
from hub.util.remove_cache import get_base_storage, get_dataset_with_zero_size_cache
from hub.util.transform import (
    check_transform_data_in,
    check_transform_ds_out,
    store_data_slice,
)
from hub.util.encoder import merge_all_chunk_id_encoders, merge_all_tensor_metas
from hub.util.exceptions import (
    TransformComposeEmptyListError,
    TransformComposeIncompatibleFunction,
)


class TransformFunction:
    def __init__(self, func, args, kwargs):
        """Creates a TransformFunction object that can be evaluated using .eval or used as a part of a Pipeline."""
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def eval(
        self,
        data_in,
        ds_out: hub.core.dataset.Dataset,
        num_workers: int = 0,
        scheduler: str = "threaded",
    ):
        """Evaluates the TransformFunction on data_in to produce an output dataset ds_out.

        Args:
            data_in: Input passed to the transform to generate output dataset. Should support __getitem__ and __len__. Can be a Hub dataset.
            ds_out (Dataset): The dataset object to which the transform will get written.
                Should have all keys being generated in output already present as tensors. It's initial state should be either:-
                - Empty i.e. all tensors have no samples. In this case all samples are added to the dataset.
                - All tensors are populated and have sampe length. In this case new samples are appended to the dataset.
            num_workers (int): The number of workers to use for performing the transform. Defaults to 0. When set to 0, it will always use serial processing, irrespective of the scheduler.
            scheduler (str): The scheduler to be used to compute the transformation. Supported values include: "serial", 'threaded' and 'processed'.

        Raises:
            InvalidInputDataError: If data_in passed to transform is invalid. It should support __getitem__ and __len__ operations. Using scheduler other than "threaded" with hub dataset having base storage as memory as data_in will also raise this.
            InvalidOutputDatasetError: If all the tensors of ds_out passed to transform don't have the same length. Using scheduler other than "threaded" with hub dataset having base storage as memory as ds_out will also raise this.
            TensorMismatchError: If one or more of the outputs generated during transform contain different tensors than the ones present in 'ds_out' provided to transform.
            UnsupportedSchedulerError: If the scheduler passed is not recognized. Supported values include: "serial", 'threaded' and 'processed'.
        """
        pipeline = Pipeline([self])
        pipeline.eval(data_in, ds_out, num_workers, scheduler)


class Pipeline:
    def __init__(self, transform_functions: List[TransformFunction]):
        """Takes a list of transform functions and creates a pipeline out of them that can be evaluated using .eval"""
        self.transform_functions = transform_functions

    def __len__(self):
        return len(self.transform_functions)

    def eval(
        self,
        data_in,
        ds_out: hub.core.dataset.Dataset,
        num_workers: int = 0,
        scheduler: str = "threaded",
    ):
        """Evaluates the pipeline on data_in to produce an output dataset ds_out.

        Args:
            data_in: Input passed to the transform to generate output dataset. Should support __getitem__ and __len__. Can be a Hub dataset.
            ds_out (Dataset): The dataset object to which the transform will get written.
                Should have all keys being generated in output already present as tensors. It's initial state should be either:-
                - Empty i.e. all tensors have no samples. In this case all samples are added to the dataset.
                - All tensors are populated and have sampe length. In this case new samples are appended to the dataset.
            num_workers (int): The number of workers to use for performing the transform. Defaults to 0. When set to 0, it will always use serial processing, irrespective of the scheduler.
            scheduler (str): The scheduler to be used to compute the transformation. Supported values include: "serial", 'threaded' and 'processed'.

        Raises:
            InvalidInputDataError: If data_in passed to transform is invalid. It should support __getitem__ and __len__ operations. Using scheduler other than "threaded" with hub dataset having base storage as memory as data_in will also raise this.
            InvalidOutputDatasetError: If all the tensors of ds_out passed to transform don't have the same length. Using scheduler other than "threaded" with hub dataset having base storage as memory as ds_out will also raise this.
            TensorMismatchError: If one or more of the outputs generated during transform contain different tensors than the ones present in 'ds_out' provided to transform.
            UnsupportedSchedulerError: If the scheduler passed is not recognized. Supported values include: "serial", 'threaded' and 'processed'.
        """
        num_workers = max(num_workers, 0)
        if num_workers == 0:
            scheduler = "serial"

        if isinstance(data_in, hub.core.dataset.Dataset):
            data_in = get_dataset_with_zero_size_cache(data_in)

        check_transform_data_in(data_in, scheduler)
        check_transform_ds_out(ds_out, scheduler)

        ds_out.flush()
        initial_autoflush = ds_out.storage.autoflush
        ds_out.storage.autoflush = False

        tensors = list(ds_out.tensors)
        compute_provider = get_compute_provider(scheduler, num_workers)

        self.run(data_in, ds_out, tensors, compute_provider, num_workers)
        ds_out.storage.autoflush = initial_autoflush

    def run(
        self,
        data_in,
        ds_out: hub.core.dataset.Dataset,
        tensors: List[str],
        compute: ComputeProvider,
        num_workers: int,
    ):
        """Runs the pipeline on the input data to produce output samples and stores in the dataset.
        This receives arguments processed and sanitized by the Pipeline.eval method.
        """
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
    """Takes a list of transform functions and creates a pipeline out of them that can be evaluated using .eval"""
    if not transform_functions:
        raise TransformComposeEmptyListError
    for index, fn in enumerate(transform_functions):
        if not isinstance(fn, TransformFunction):
            raise TransformComposeIncompatibleFunction(index)
    return Pipeline(transform_functions)


def compute(fn):
    """Compute is a decorator for functions.
    The functions should have atleast 2 argument, the first two will correspond to sample_in and samples_out.
    There can be as many other arguments as required.
    The output should be appended/extended to the second argument in a hub like syntax.
    Any value returned by the fn will be ignored.

    Example:
    @hub.compute
    def your_function(sample_in: Any, samples_out, your_arg0, your_arg1=0):
        samples_out.your_tensor.append(your_arg0 * your_arg1)
    """

    def inner(*args, **kwargs):
        return TransformFunction(fn, args, kwargs)

    return inner
