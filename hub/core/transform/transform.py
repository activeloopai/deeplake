from uuid import uuid4
import hub
from typing import Callable, List, Optional
from itertools import repeat
from hub.core.compute.provider import ComputeProvider
from hub.core.storage.memory import MemoryProvider
from hub.util.bugout_reporter import hub_reporter
from hub.util.compute import get_compute_provider
from hub.util.dataset import try_flushing
from hub.util.remove_cache import get_base_storage, get_dataset_with_zero_size_cache
from hub.util.transform import (
    check_lengths,
    check_transform_data_in,
    check_transform_ds_out,
    create_slices,
    delete_overwritten_chunks,
    get_lengths_generated,
    get_old_chunk_paths,
    get_pbar_description,
    process_transform_result,
    sanitize_workers_scheduler,
    store_data_slice,
    store_data_slice_with_pbar,
)
from hub.util.encoder import merge_all_meta_info
from hub.util.exceptions import (
    HubComposeEmptyListError,
    HubComposeIncompatibleFunction,
    TransformError,
)
from hub.hooks import dataset_written, dataset_read
from hub.util.version_control import auto_checkout, load_meta
from hub.util.class_label import sync_labels
import numpy as np


class ComputeFunction:
    def __init__(self, func, args, kwargs, name: Optional[str] = None):
        """Creates a ComputeFunction object that can be evaluated using .eval or used as a part of a Pipeline."""
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.name = self.func.__name__ if name is None else name

    def eval(
        self,
        data_in,
        ds_out: Optional[hub.Dataset] = None,
        num_workers: int = 0,
        scheduler: str = "threaded",
        progressbar: bool = True,
        skip_ok: bool = False,
        check_lengths: bool = True,
        pad_data_in: bool = False,
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
            check_lengths (bool): If True, checks whether ds_out has tensors of same lengths initially.
            pad_data_in (bool): NOTE: This is only applicable if data_in is a Hub dataset. If True, pads tensors of data_in to match the length of the largest tensor in data_in.
                Defaults to False.

        Raises:
            InvalidInputDataError: If data_in passed to transform is invalid. It should support \__getitem__ and \__len__ operations. Using scheduler other than "threaded" with hub dataset having base storage as memory as data_in will also raise this.
            InvalidOutputDatasetError: If all the tensors of ds_out passed to transform don't have the same length. Using scheduler other than "threaded" with hub dataset having base storage as memory as ds_out will also raise this.
            TensorMismatchError: If one or more of the outputs generated during transform contain different tensors than the ones present in 'ds_out' provided to transform.
            UnsupportedSchedulerError: If the scheduler passed is not recognized. Supported values include: 'serial', 'threaded', 'processed' and 'ray'.
        """

        pipeline = Pipeline([self])
        pipeline.eval(
            data_in,
            ds_out,
            num_workers,
            scheduler,
            progressbar,
            skip_ok,
            check_lengths,
            pad_data_in,
        )

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
        check_lengths: bool = True,
        pad_data_in: bool = False,
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
            check_lengths (bool): If True, checks whether ds_out has tensors of same lengths initially.
            pad_data_in (bool): NOTE: This is only applicable if data_in is a Hub dataset. If True, pads tensors of data_in to match the length of the largest tensor in data_in.
                Defaults to False.

        Raises:
            InvalidInputDataError: If data_in passed to transform is invalid. It should support \__getitem__ and \__len__ operations. Using scheduler other than "threaded" with hub dataset having base storage as memory as data_in will also raise this.
            InvalidOutputDatasetError: If all the tensors of ds_out passed to transform don't have the same length. Using scheduler other than "threaded" with hub dataset having base storage as memory as ds_out will also raise this.
            TensorMismatchError: If one or more of the outputs generated during transform contain different tensors than the ones present in 'ds_out' provided to transform.
            UnsupportedSchedulerError: If the scheduler passed is not recognized. Supported values include: 'serial', 'threaded', 'processed' and 'ray'.
            TransformError: All other exceptions raised if there are problems while running the pipeline.
        """
        num_workers, scheduler = sanitize_workers_scheduler(num_workers, scheduler)
        overwrite = ds_out is None
        hub_reporter.feature_report(
            feature_name="eval",
            parameters={"Num_Workers": str(num_workers), "Scheduler": scheduler},
        )
        check_transform_data_in(data_in, scheduler)

        if isinstance(data_in, hub.Dataset):
            try_flushing(data_in)
            if overwrite:
                auto_checkout(data_in)
            original_data_in = data_in
            data_in = get_dataset_with_zero_size_cache(data_in)
            if pad_data_in:
                initial_padding_state = data_in._pad_tensors
                data_in._enable_padding()

        target_ds = data_in if overwrite else ds_out

        check_transform_ds_out(target_ds, scheduler, check_lengths)

        # if overwrite then we've already flushed and autocheckecked out data_in which is target_ds now
        if not overwrite:
            target_ds.flush()
            auto_checkout(target_ds)

        compute_provider = get_compute_provider(scheduler, num_workers)
        compute_id = str(uuid4().hex)
        target_ds._send_compute_progress(compute_id=compute_id, start=True, progress=0)

        initial_autoflush = target_ds.storage.autoflush
        target_ds.storage.autoflush = False
        progress_end_args = {"compute_id": compute_id, "progress": 100, "end": True}

        if not check_lengths:
            skip_ok = True

        try:
            self.run(
                data_in,
                target_ds,
                compute_provider,
                num_workers,
                scheduler,
                progressbar,
                overwrite,
                skip_ok,
            )
            target_ds._send_compute_progress(**progress_end_args, status="success")
        except Exception as e:
            target_ds._send_compute_progress(**progress_end_args, status="failed")
            raise TransformError(e) from e
        finally:
            compute_provider.close()
            if overwrite:
                original_data_in.storage.clear_cache_without_flush()
                load_meta(original_data_in)
                if pad_data_in and not initial_padding_state:
                    original_data_in._disable_padding()
            else:
                load_meta(target_ds)
                target_ds.storage.autoflush = initial_autoflush

    def run(
        self,
        data_in,
        target_ds: hub.Dataset,
        compute: ComputeProvider,
        num_workers: int,
        scheduler: str,
        progressbar: bool = True,
        overwrite: bool = False,
        skip_ok: bool = False,
    ):
        """Runs the pipeline on the input data to produce output samples and stores in the dataset.
        This receives arguments processed and sanitized by the Pipeline.eval method.
        """
        if isinstance(data_in, hub.Dataset):
            dataset_read(data_in)
        slices = create_slices(data_in, num_workers)
        storage = get_base_storage(target_ds.storage)
        class_label_tensors = [
            tensor.key
            for tensor in target_ds.tensors.values()
            if tensor.base_htype == "class_label"
            and not tensor.meta._disable_temp_transform
        ]
        label_temp_tensors = {}
        actual_tensors = (
            None
            if not class_label_tensors
            else [target_ds[t].key for t in target_ds.tensors]
        )

        for tensor in class_label_tensors:
            temp_tensor = f"_{tensor}_{uuid4().hex[:4]}"
            with target_ds:
                temp_tensor_obj = target_ds.create_tensor(
                    temp_tensor,
                    htype="class_label",
                    create_sample_info_tensor=False,
                    create_shape_tensor=False,
                    create_id_tensor=False,
                )
                temp_tensor_obj.meta._disable_temp_transform = True
                label_temp_tensors[tensor] = temp_tensor
            target_ds.flush()

        visible_tensors = list(target_ds.tensors)
        visible_tensors = [target_ds[t].key for t in visible_tensors]
        visible_tensors = list(set(visible_tensors) - set(class_label_tensors))

        tensors = list(target_ds._tensors())
        tensors = [target_ds[t].key for t in tensors]
        tensors = list(set(tensors) - set(class_label_tensors))

        group_index = target_ds.group_index
        version_state = target_ds.version_state
        if isinstance(storage, MemoryProvider):
            storages = [storage] * len(slices)
        else:
            storages = [storage.copy() for _ in slices]
        args = (
            group_index,
            tensors,
            visible_tensors,
            label_temp_tensors,
            actual_tensors,
            self,
            version_state,
            target_ds.link_creds,
            skip_ok,
        )
        map_inp = zip(slices, storages, repeat(args))

        if progressbar:
            desc = get_pbar_description(self.functions)
            result = compute.map_with_progressbar(
                store_data_slice_with_pbar,
                map_inp,
                total_length=len(data_in),
                desc=desc,
            )
        else:
            result = compute.map(store_data_slice, map_inp)
        result = process_transform_result(result)

        all_num_samples, all_tensors_generated_length = get_lengths_generated(
            result["tensor_metas"], tensors
        )

        check_lengths(all_tensors_generated_length, skip_ok)

        generated_tensors = [
            tensor for tensor, l in all_tensors_generated_length.items() if l > 0
        ]

        old_chunk_paths = get_old_chunk_paths(target_ds, generated_tensors, overwrite)
        merge_all_meta_info(
            target_ds, storage, generated_tensors, overwrite, all_num_samples, result
        )
        delete_overwritten_chunks(old_chunk_paths, storage, overwrite)
        dataset_written(target_ds)

        if label_temp_tensors:
            sync_labels(
                target_ds,
                label_temp_tensors,
                result["hash_label_maps"],
                num_workers=num_workers,
                scheduler=scheduler,
                verbose=progressbar,
            )


def compose(functions: List[ComputeFunction]):  # noqa: DAR101, DAR102, DAR201, DAR401
    """Takes a list of functions decorated using :func:`hub.compute` and creates a pipeline that can be evaluated using .eval

    Example::

        pipeline = hub.compose([my_fn(a=3), another_function(b=2)])
        pipeline.eval(data_in, ds_out, scheduler="processed", num_workers=2)

    The ``eval`` method evaluates the pipeline/transform function.

    It has the following arguments:

    - ``data_in``: Input passed to the transform to generate output dataset.

        - It should support ``__getitem__`` and ``__len__``. This can be a Hub dataset.

    - ``ds_out (Dataset, optional)``: The dataset object to which the transform will get written.

        - If this is not provided, data_in will be overwritten if it is a Hub dataset, otherwise error will be raised.
        - It should have all keys being generated in output already present as tensors.
        - It's initial state should be either:

            - Empty i.e. all tensors have no samples. In this case all samples are added to the dataset.
            - All tensors are populated and have same length. In this case new samples are appended to the dataset.

    - ``num_workers (int)``: The number of workers to use for performing the transform.

        - Defaults to 0. When set to 0, it will always use serial processing, irrespective of the scheduler.

    - ``scheduler (str)``: The scheduler to be used to compute the transformation.

        - Supported values include: 'serial', 'threaded', 'processed' and 'ray'. Defaults to 'threaded'.

    - ``progressbar (bool)``: Displays a progress bar if True (default).

    - ``skip_ok (bool)``: If True, skips the check for output tensors generated.

        - This allows the user to skip certain tensors in the function definition.
        - This is especially useful for inplace transformations in which certain tensors are not modified. Defaults to ``False``.

    It raises the following errors:

    - ``InvalidInputDataError``: If data_in passed to transform is invalid. It should support ``__getitem__`` and ``__len__`` operations. Using scheduler other than "threaded" with hub dataset having base storage as memory as data_in will also raise this.

    - ``InvalidOutputDatasetError``: If all the tensors of ds_out passed to transform don't have the same length. Using scheduler other than "threaded" with hub dataset having base storage as memory as ds_out will also raise this.

    - ``TensorMismatchError``: If one or more of the outputs generated during transform contain different tensors than the ones present in 'ds_out' provided to transform.

    - ``UnsupportedSchedulerError``: If the scheduler passed is not recognized. Supported values include: 'serial', 'threaded', 'processed' and 'ray'.

    - ``TransformError``: All other exceptions raised if there are problems while running the pipeline.
    """
    if not functions:
        raise HubComposeEmptyListError
    for index, fn in enumerate(functions):
        if not isinstance(fn, ComputeFunction):
            raise HubComposeIncompatibleFunction(index)
    return Pipeline(functions)


def compute(
    fn,
    name: Optional[str] = None,
) -> Callable[..., ComputeFunction]:  # noqa: DAR101, DAR102, DAR201, DAR401
    """Compute is a decorator for functions.

    The functions should have atleast 2 argument, the first two will correspond to ``sample_in`` and ``samples_out``.

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

    The ``eval`` method evaluates the pipeline/transform function.

    It has the following arguments:

    - ``data_in``: Input passed to the transform to generate output dataset.

        - It should support ``__getitem__`` and ``__len__``. This can be a Hub dataset.

    - ``ds_out (Dataset, optional)``: The dataset object to which the transform will get written.

        - If this is not provided, data_in will be overwritten if it is a Hub dataset, otherwise error will be raised.
        - It should have all keys being generated in output already present as tensors.
        - It's initial state should be either:

            - Empty i.e. all tensors have no samples. In this case all samples are added to the dataset.
            - All tensors are populated and have same length. In this case new samples are appended to the dataset.

    - ``num_workers (int)``: The number of workers to use for performing the transform.

        - Defaults to 0. When set to 0, it will always use serial processing, irrespective of the scheduler.

    - ``scheduler (str)``: The scheduler to be used to compute the transformation.

        - Supported values include: 'serial', 'threaded', 'processed' and 'ray'. Defaults to 'threaded'.

    - ``progressbar (bool)``: Displays a progress bar if True (default).

    - ``skip_ok (bool)``: If True, skips the check for output tensors generated.

        - This allows the user to skip certain tensors in the function definition.
        - This is especially useful for inplace transformations in which certain tensors are not modified. Defaults to ``False``.

    It raises the following errors:

    - ``InvalidInputDataError``: If data_in passed to transform is invalid. It should support ``__getitem__`` and ``__len__`` operations. Using scheduler other than "threaded" with hub dataset having base storage as memory as data_in will also raise this.

    - ``InvalidOutputDatasetError``: If all the tensors of ds_out passed to transform don't have the same length. Using scheduler other than "threaded" with hub dataset having base storage as memory as ds_out will also raise this.

    - ``TensorMismatchError``: If one or more of the outputs generated during transform contain different tensors than the ones present in 'ds_out' provided to transform.

    - ``UnsupportedSchedulerError``: If the scheduler passed is not recognized. Supported values include: 'serial', 'threaded', 'processed' and 'ray'.

    - ``TransformError``: All other exceptions raised if there are problems while running the pipeline.
    """

    def inner(*args, **kwargs):
        return ComputeFunction(fn, args, kwargs, name)

    return inner
