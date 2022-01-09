from uuid import uuid4
import hub
import math
from typing import Callable, List, Optional
from itertools import repeat
import warnings
from hub.constants import FIRST_COMMIT_ID
from hub.core.compute.provider import ComputeProvider
from hub.util.bugout_reporter import hub_reporter
from hub.util.compute import get_compute_provider
from hub.util.remove_cache import get_base_storage, get_dataset_with_zero_size_cache
from hub.util.transform import (
    check_transform_data_in,
    check_transform_ds_out,
    get_pbar_description,
    store_data_slice,
    store_data_slice_with_pbar,
)
from hub.util.cachable import reset_cachables
from hub.util.encoder import (
    merge_all_chunk_id_encoders,
    merge_all_commit_diffs,
    merge_all_tensor_metas,
    merge_all_tile_encoders,
    merge_all_commit_chunk_sets,
)
from hub.util.exceptions import (
    HubComposeEmptyListError,
    HubComposeIncompatibleFunction,
    TransformError,
)
from hub.util.version_control import auto_checkout


class ComputeFunction:
    def __init__(self, func, args, kwargs):
        """Creates a ComputeFunction object that can be evaluated using .eval or used as a part of a Pipeline."""
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def eval(
        self,
        data_in,
        ds_out: Optional[hub.Dataset] = None,
        num_workers: int = 0,
        scheduler: str = "threaded",
        progressbar: bool = True,
        skip_ok: bool = False,
    ):
        """Evaluates the ComputeFunction on data_in to produce an output dataset ds_out.

        Args:
            data_in: Input passed to the transform to generate output dataset. Should support \__getitem__ and \__len__. Can be a Hub dataset.
            ds_out (Dataset, optional): The dataset object to which the transform will get written. If this is not provided, data_in will be overwritten if it is a Hub dataset, otherwise error will be raised.
                It should have all keys being generated in output already present as tensors. It's initial state should be either:-
                - Empty i.e. all tensors have no samples. In this case all samples are added to the dataset.
                - All tensors are populated and have sampe length. In this case new samples are appended to the dataset.
            num_workers (int): The number of workers to use for performing the transform. Defaults to 0. When set to 0, it will always use serial processing, irrespective of the scheduler.
            scheduler (str): The scheduler to be used to compute the transformation. Supported values include: 'serial', 'threaded', 'processed' and 'ray'.
                Defaults to 'threaded'.
            progressbar (bool): Displays a progress bar if True (default).
            skip_ok (bool): If True, skips the check for output tensors generated. This allows the user to skip certain tensors in the function definition.
                This is especially useful for inplace transformations in which certain tensors are not modified. Defaults to False.


        Raises:
            InvalidInputDataError: If data_in passed to transform is invalid. It should support \__getitem__ and \__len__ operations. Using scheduler other than "threaded" with hub dataset having base storage as memory as data_in will also raise this.
            InvalidOutputDatasetError: If all the tensors of ds_out passed to transform don't have the same length. Using scheduler other than "threaded" with hub dataset having base storage as memory as ds_out will also raise this.
            TensorMismatchError: If one or more of the outputs generated during transform contain different tensors than the ones present in 'ds_out' provided to transform.
            UnsupportedSchedulerError: If the scheduler passed is not recognized. Supported values include: 'serial', 'threaded', 'processed' and 'ray'.
        """

        pipeline = Pipeline([self])
        pipeline.eval(data_in, ds_out, num_workers, scheduler, progressbar, skip_ok)

    def __call__(self, sample_in):
        return self.func(sample_in, *self.args, **self.kwargs)


class Pipeline:
    def __init__(self, functions: List[ComputeFunction]):
        """Takes a list of functions decorated using hub.compute and creates a pipeline that can be evaluated using .eval"""
        self.functions = functions

    def __len__(self):
        return len(self.functions)

    def eval(
        self,
        data_in,
        ds_out: Optional[hub.Dataset] = None,
        num_workers: int = 0,
        scheduler: str = "threaded",
        progressbar: bool = True,
        skip_ok: bool = False,
    ):
        """Evaluates the pipeline on data_in to produce an output dataset ds_out.

        Args:
            data_in: Input passed to the transform to generate output dataset. Should support \__getitem__ and \__len__. Can be a Hub dataset.
            ds_out (Dataset, optional): The dataset object to which the transform will get written. If this is not provided, data_in will be overwritten if it is a Hub dataset, otherwise error will be raised.
                It should have all keys being generated in output already present as tensors. It's initial state should be either:-
                - Empty i.e. all tensors have no samples. In this case all samples are added to the dataset.
                - All tensors are populated and have sampe length. In this case new samples are appended to the dataset.
            num_workers (int): The number of workers to use for performing the transform. Defaults to 0. When set to 0, it will always use serial processing, irrespective of the scheduler.
            scheduler (str): The scheduler to be used to compute the transformation. Supported values include: 'serial', 'threaded', 'processed' and 'ray'.
                Defaults to 'threaded'.
            progressbar (bool): Displays a progress bar if True (default).
            skip_ok (bool): If True, skips the check for output tensors generated. This allows the user to skip certain tensors in the function definition.
                This is especially useful for inplace transformations in which certain tensors are not modified. Defaults to False.

        Raises:
            InvalidInputDataError: If data_in passed to transform is invalid. It should support \__getitem__ and \__len__ operations. Using scheduler other than "threaded" with hub dataset having base storage as memory as data_in will also raise this.
            InvalidOutputDatasetError: If all the tensors of ds_out passed to transform don't have the same length. Using scheduler other than "threaded" with hub dataset having base storage as memory as ds_out will also raise this.
            TensorMismatchError: If one or more of the outputs generated during transform contain different tensors than the ones present in 'ds_out' provided to transform.
            UnsupportedSchedulerError: If the scheduler passed is not recognized. Supported values include: 'serial', 'threaded', 'processed' and 'ray'.
            TransformError: All other exceptions raised if there are problems while running the pipeline.
        """
        if num_workers <= 0:
            scheduler = "serial"
        num_workers = max(num_workers, 1)
        original_data_in = data_in
        if isinstance(data_in, hub.Dataset):
            data_in = get_dataset_with_zero_size_cache(data_in)

        hub_reporter.feature_report(
            feature_name="eval",
            parameters={"Num_Workers": str(num_workers), "Scheduler": scheduler},
        )

        check_transform_data_in(data_in, scheduler)
        target_ds = data_in if ds_out is None else ds_out
        check_transform_ds_out(target_ds, scheduler)

        initial_autoflush = target_ds.storage.autoflush
        target_ds.storage.autoflush = False

        # if it is None, then we've already flushed data_in which is target_ds now
        if ds_out is not None:
            target_ds.flush()

        # if not the head node, checkout to an auto branch that is newly created
        auto_checkout(target_ds)

        overwrite = ds_out is None
        if overwrite:
            original_data_in.clear_cache()

        compute_provider = get_compute_provider(scheduler, num_workers)

        compute_id = str(uuid4().hex)
        target_ds._send_compute_progress(compute_id=compute_id, start=True, progress=0)
        try:
            self.run(
                data_in,
                target_ds,
                compute_provider,
                num_workers,
                progressbar,
                overwrite,
                skip_ok,
            )
            target_ds._send_compute_progress(
                compute_id=compute_id, end=True, progress=100, status="success"
            )
        except Exception as e:
            target_ds._send_compute_progress(
                compute_id=compute_id, end=True, progress=100, status="failed"
            )
            raise TransformError(e)
        finally:
            compute_provider.close()
            target_ds.storage.autoflush = initial_autoflush

    def run(
        self,
        data_in,
        target_ds: hub.Dataset,
        compute: ComputeProvider,
        num_workers: int,
        progressbar: bool = True,
        overwrite: bool = False,
        skip_ok: bool = False,
    ):
        """Runs the pipeline on the input data to produce output samples and stores in the dataset.
        This receives arguments processed and sanitized by the Pipeline.eval method.
        """
        size = math.ceil(len(data_in) / num_workers)
        slices = [data_in[i * size : (i + 1) * size] for i in range(num_workers)]
        storage = get_base_storage(target_ds.storage)
        group_index = target_ds.group_index
        version_state = target_ds.version_state

        tensors = list(target_ds.tensors)
        tensors = [target_ds.tensors[t].key for t in tensors]
        map_inp = zip(
            slices,
            repeat((storage, group_index, tensors, self, version_state, skip_ok)),
        )
        if progressbar:
            desc = get_pbar_description(self.functions)
            metas_and_encoders = compute.map_with_progressbar(
                store_data_slice_with_pbar,
                map_inp,
                total_length=len(data_in),
                desc=desc,
            )
        else:
            metas_and_encoders = compute.map(store_data_slice, map_inp)

        (
            all_tensor_metas,
            all_chunk_id_encoders,
            all_tile_encoders,
            all_chunk_commit_sets,
            all_commit_diffs,
        ) = zip(*metas_and_encoders)
        all_num_samples = []
        all_tensors_generated_length = {tensor: 0 for tensor in tensors}
        for tensor_meta_dict in all_tensor_metas:
            num_samples_dict = {}
            for tensor, meta in tensor_meta_dict.items():
                all_tensors_generated_length[tensor] += meta.length
                num_samples_dict[tensor] = meta.length
            all_num_samples.append(num_samples_dict)
        first_length = None
        if skip_ok:
            for tensor, length in all_tensors_generated_length.items():
                if first_length is None:
                    first_length = length
                elif length not in [0, first_length]:
                    warnings.warn(
                        "Length of all tensors generated is not the same, this may lead to unexpected behavior."
                    )
                    break

        generated_tensors = [
            tensor
            for tensor, length in all_tensors_generated_length.items()
            if length > 0
        ]

        if overwrite:
            for key, tensor in target_ds.tensors.items():
                if key in generated_tensors:
                    storage.delete_multiple(tensor.chunk_engine.list_all_chunks_path())
        merge_all_commit_diffs(
            all_commit_diffs, target_ds, storage, overwrite, generated_tensors
        )
        merge_all_tile_encoders(
            all_tile_encoders,
            all_num_samples,
            target_ds,
            storage,
            overwrite,
            generated_tensors,
        )
        merge_all_tensor_metas(
            all_tensor_metas, target_ds, storage, overwrite, generated_tensors
        )
        merge_all_chunk_id_encoders(
            all_chunk_id_encoders, target_ds, storage, overwrite, generated_tensors
        )
        if target_ds.commit_id is not None:
            merge_all_commit_chunk_sets(
                all_chunk_commit_sets, target_ds, storage, overwrite, generated_tensors
            )

        reset_cachables(target_ds, generated_tensors)


def compose(functions: List[ComputeFunction]):  # noqa: DAR101, DAR102, DAR201, DAR401
    """Takes a list of functions decorated using hub.compute and creates a pipeline that can be evaluated using .eval

    Example::

        pipeline = hub.compose([my_fn(a=3), another_function(b=2)])
        pipeline.eval(data_in, ds_out, scheduler="processed", num_workers=2)

    The __eval__ method evaluates the pipeline/transform function.

    It has the following arguments:-

    - data_in: Input passed to the transform to generate output dataset.
    It should support \__getitem__ and \__len__. This can be a Hub dataset.
    - ds_out (Dataset, optional): The dataset object to which the transform will get written.
    If this is not provided, data_in will be overwritten if it is a Hub dataset, otherwise error will be raised.
    It should have all keys being generated in output already present as tensors.
    It's initial state should be either:-
        - Empty i.e. all tensors have no samples. In this case all samples are added to the dataset.
        - All tensors are populated and have sampe length. In this case new samples are appended to the dataset.
    - num_workers (int): The number of workers to use for performing the transform.
    Defaults to 0. When set to 0, it will always use serial processing, irrespective of the scheduler.
    - scheduler (str): The scheduler to be used to compute the transformation.
    Supported values include: 'serial', 'threaded', 'processed' and 'ray'. Defaults to 'threaded'.
    - progressbar (bool): Displays a progress bar if True (default).
    - skip_ok (bool): If True, skips the check for output tensors generated. This allows the user to skip certain tensors in the function definition.
    This is especially useful for inplace transformations in which certain tensors are not modified. Defaults to False.

    It raises the following errors:-

    - InvalidInputDataError: If data_in passed to transform is invalid. It should support \__getitem__ and \__len__ operations. Using scheduler other than "threaded" with hub dataset having base storage as memory as data_in will also raise this.
    - InvalidOutputDatasetError: If all the tensors of ds_out passed to transform don't have the same length. Using scheduler other than "threaded" with hub dataset having base storage as memory as ds_out will also raise this.
    - TensorMismatchError: If one or more of the outputs generated during transform contain different tensors than the ones present in 'ds_out' provided to transform.
    - UnsupportedSchedulerError: If the scheduler passed is not recognized. Supported values include: 'serial', 'threaded', 'processed' and 'ray'.
    - TransformError: All other exceptions raised if there are problems while running the pipeline.
    """
    if not functions:
        raise HubComposeEmptyListError
    for index, fn in enumerate(functions):
        if not isinstance(fn, ComputeFunction):
            raise HubComposeIncompatibleFunction(index)
    return Pipeline(functions)


def compute(
    fn,
) -> Callable[..., ComputeFunction]:  # noqa: DAR101, DAR102, DAR201, DAR401
    """Compute is a decorator for functions.
    The functions should have atleast 2 argument, the first two will correspond to sample_in and samples_out.
    There can be as many other arguments as required.
    The output should be appended/extended to the second argument in a hub like syntax.
    Any value returned by the fn will be ignored.

    Example::

        @hub.compute
        def my_fn(sample_in: Any, samples_out, my_arg0, my_arg1=0):
            samples_out.my_tensor.append(my_arg0 * my_arg1)

        # This transform can be used using the eval method in one of these 2 ways:-

        # Directly evaluating the method
        # here arg0 and arg1 correspond to the 3rd and 4th argument in my_fn
        my_fn(arg0, arg1).eval(data_in, ds_out, scheduler="threaded", num_workers=5)

        # As a part of a Transform pipeline containing other functions
        pipeline = hub.compose([my_fn(a, b), another_function(x=2)])
        pipeline.eval(data_in, ds_out, scheduler="processed", num_workers=2)

    The __eval__ method evaluates the pipeline/transform function.

    It has the following arguments:-

    - data_in: Input passed to the transform to generate output dataset.
    It should support \__getitem__ and \__len__. This can be a Hub dataset.
    - ds_out (Dataset, optional): The dataset object to which the transform will get written.
    If this is not provided, data_in will be overwritten if it is a Hub dataset, otherwise error will be raised.
    It should have all keys being generated in output already present as tensors.
    It's initial state should be either:-
        - Empty i.e. all tensors have no samples. In this case all samples are added to the dataset.
        - All tensors are populated and have sampe length. In this case new samples are appended to the dataset.
    - num_workers (int): The number of workers to use for performing the transform.
    Defaults to 0. When set to 0, it will always use serial processing, irrespective of the scheduler.
    - scheduler (str): The scheduler to be used to compute the transformation.
    Supported values include: 'serial', 'threaded', 'processed' and 'ray'. Defaults to 'threaded'.
    - progressbar (bool): Displays a progress bar if True (default).
    - skip_ok (bool): If True, skips the check for output tensors generated. This allows the user to skip certain tensors in the function definition.
    This is especially useful for inplace transformations in which certain tensors are not modified. Defaults to False.

    It raises the following errors:-

    - InvalidInputDataError: If data_in passed to transform is invalid. It should support \__getitem__ and \__len__ operations. Using scheduler other than "threaded" with hub dataset having base storage as memory as data_in will also raise this.
    - InvalidOutputDatasetError: If all the tensors of ds_out passed to transform don't have the same length. Using scheduler other than "threaded" with hub dataset having base storage as memory as ds_out will also raise this.
    - TensorMismatchError: If one or more of the outputs generated during transform contain different tensors than the ones present in 'ds_out' provided to transform.
    - UnsupportedSchedulerError: If the scheduler passed is not recognized. Supported values include: 'serial', 'threaded', 'processed' and 'ray'.
    - TransformError: All other exceptions raised if there are problems while running the pipeline.
    """

    def inner(*args, **kwargs):
        return ComputeFunction(fn, args, kwargs)

    return inner
