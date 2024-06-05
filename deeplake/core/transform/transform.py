from uuid import uuid4
import deeplake
from typing import Callable, List, Optional
from itertools import repeat
from deeplake.core.compute.provider import ComputeProvider, get_progress_bar
from deeplake.core.storage.memory import MemoryProvider
from deeplake.util.bugout_reporter import deeplake_reporter
from deeplake.util.compute import get_compute_provider
from deeplake.util.path import relpath
from deeplake.util.remove_cache import get_base_storage
from deeplake.util.transform import (
    check_lengths,
    check_transform_data_in,
    check_transform_ds_out,
    close_states,
    create_slices,
    delete_overwritten_chunks,
    get_lengths_generated,
    get_old_chunk_paths,
    get_pbar_description,
    prepare_data_in,
    process_transform_result,
    reload_and_rechunk,
    sanitize_workers_scheduler,
    store_data_slice,
    store_data_slice_with_pbar,
    check_checkpoint_interval,
    len_data_in,
    transform_summary,
)
from deeplake.util.encoder import merge_all_meta_info
from deeplake.util.exceptions import (
    AllSamplesSkippedError,
    HubComposeEmptyListError,
    HubComposeIncompatibleFunction,
    TransformError,
)
from deeplake.hooks import dataset_written, dataset_read
from deeplake.util.version_control import auto_checkout
from deeplake.util.class_label import sync_labels
from deeplake.constants import DEFAULT_TRANSFORM_SAMPLE_CACHE_SIZE


class ComputeFunction:
    def __init__(self, func, args, kwargs, name: Optional[str] = None):
        """Creates a ComputeFunction object that can be evaluated using ``.eval()`` or used as a part of a Pipeline.
        Compute Functions are evaluated in parallel, where the input data iterable is devided between the workers for processing.
        """
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.name = self.func.__name__ if name is None else name

    def eval(
        self,
        data_in,
        ds_out: Optional[deeplake.Dataset] = None,
        num_workers: int = 0,
        scheduler: str = "threaded",
        progressbar: bool = True,
        skip_ok: bool = False,
        check_lengths: bool = True,
        pad_data_in: bool = False,
        read_only_ok: bool = False,
        cache_size: int = DEFAULT_TRANSFORM_SAMPLE_CACHE_SIZE,
        checkpoint_interval: int = 0,
        ignore_errors: bool = False,
        **kwargs,
    ):
        """
        Evaluates the ComputeFunction on ``data_in`` to produce an output dataset ``ds_out``. The purpose of compute functions is to process the input data in parallel,
        which is useful when rapidly ingesting data to a Deep Lake dataset. Compute Functions can also be executed in-place, where it modifies the input dataset (see ``ds_out`` parameters below) instead of writing to a new dataset.

        Args:
            data_in: Input passed to the transform to generate output dataset. Should support ``__getitem__`` and ``__len__`` operations. Can be a Deep Lake dataset.
            ds_out (Dataset, optional): The dataset object to which the transform will get written. If this is not provided, the ComputeFunction will operate in-place, which means that data will be written to tensors in ``data_in`` .
                All tensors modified in the ComputeFunction should already be defined in ``ds_out``. It's initial state should be either:
                - Empty i.e. all tensors have no samples. In this case all samples are added to the dataset.
                - All tensors are populated and have same length. In this case new samples are appended to the dataset.
            num_workers (int): The number of workers to use for performing the transform. Defaults to 0. When set to 0, it will always use serial processing, irrespective of the scheduler.
            scheduler (str): The scheduler to be used to compute the transformation. Supported values include: ``serial``, ``threaded``, and ``processed``.
                Defaults to ``threaded``.
            progressbar (bool): Displays a progress bar if True (default).
            skip_ok (bool): If ``True``, skips the check for output tensors generated. This allows the user to skip certain tensors in the function definition.
                This is especially useful for inplace transformations in which certain tensors are not modified. Defaults to False.
            check_lengths (bool): If ``True``, checks whether ds_out has tensors of same lengths initially.
            pad_data_in (bool): NOTE: This is only applicable if ``data_in`` is a Deep Lake dataset. If True, pads tensors of data_in to match the length of the largest tensor in data_in.
                Defaults to False.
            read_only_ok (bool): If ``True`` and output dataset is same as input dataset, the read-only check is skipped. This can be used to read data in parallel without making changes to underlying dataset.
                Defaults to False.
            cache_size (int): Cache size to be used by transform per worker.
            checkpoint_interval (int): If > 0, the ComputeFunction will be checkpointed with a commit every ``checkpoint_interval`` input samples to avoid restarting full transform due to intermitten failures. If the transform is interrupted, the intermediate data is deleted and the dataset is reset to the last commit.
                If <= 0, no checkpointing is done. Checkpoint interval should be a multiple of num_workers if ``num_workers`` > 0. Defaults to 0.
            ignore_errors (bool): If ``True``, input samples that causes transform to fail will be skipped and the errors will be ignored **if possible**.
            **kwargs: Additional arguments.

        Raises:
            InvalidInputDataError: If ``data_in`` passed to transform is invalid. It should support ``__getitem__`` and ``__len__`` operations. Using scheduler other than "threaded" with deeplake dataset having base storage as memory as data_in will also raise this.
            InvalidOutputDatasetError: If all the tensors of ds_out passed to transform don't have the same length. Using scheduler other than "threaded" with deeplake dataset having base storage as memory as ds_out will also raise this.
            TensorMismatchError: If one or more of the outputs generated during transform contain different tensors than the ones present in 'ds_out' provided to transform.
            UnsupportedSchedulerError: If the scheduler passed is not recognized. Supported values include: ``serial``, ``threaded``, and ``processed``.
            TransformError: All other exceptions raised if there are problems while running the ComputeFunction.
            ValueError: If ``num_workers`` > 0 and ``checkpoint_interval`` is not a multiple of ``num_workers`` or if ``checkpoint_interval`` > 0 and ds_out is None.
            AllSamplesSkippedError: If all samples are skipped during execution of the Pipeline.

        Example:
            # Suppose we have a list of dictionaries that we want to upload in parallel to a Deep Lake dataset.
            data_in = [{"label": "cat", "score": 0.9}, {"label": "dog", "score": 0.8}]

            # First, we define a function that takes a single element of the `data_in` list and uploads it to a Deep Lake dataset
            @deeplake.compute
            def data_upload(data_dict_in, sample_out, score_multiplier):
                label = data_dict_in["label"]
                score = data_dict_in["score"] * score_multiplier
                sample_out.append({"label": label, "score": score})

            # To evaluate the function, static parameters (if any) are specified to the function input, and the input data iterable and output dataset are specified to `.eval(...)`
            data_upload(score_multiplier).eval(data_in, ds_out, scheduler="threaded", num_workers=4)

        Note:
            ``pad_data_in`` is only applicable if ``data_in`` is a Deep Lake dataset.
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
            read_only_ok,
            cache_size,
            checkpoint_interval,
            ignore_errors,
            **kwargs,
        )

    def __call__(self, sample_in):
        return self.func(sample_in, *self.args, **self.kwargs)


class Pipeline:
    def __init__(self, functions: List[ComputeFunction]):
        """Takes a list of functions decorated using :func:`deeplake.compute` and creates a pipeline that can be evaluated using ``.eval()``"""
        self.functions = functions

    def __len__(self):
        return len(self.functions)

    def eval(
        self,
        data_in,
        ds_out: Optional[deeplake.Dataset] = None,
        num_workers: int = 0,
        scheduler: str = "threaded",
        progressbar: bool = True,
        skip_ok: bool = False,
        check_lengths: bool = True,
        pad_data_in: bool = False,
        read_only_ok: bool = False,
        cache_size: int = DEFAULT_TRANSFORM_SAMPLE_CACHE_SIZE,
        checkpoint_interval: int = 0,
        ignore_errors: bool = False,
        verbose: bool = True,
        **kwargs,
    ):
        """
        Evaluates the Pipeline of ComputeFunctions on ``data_in`` to produce an output dataset ``ds_out``. The purpose of compute functions is to process the input data in parallel,
        which is useful when rapidly ingesting data to a Deep Lake dataset. Pipelines can also be executed in-place, where it modifies the input dataset (see ``ds_out`` parameters below) instead of writing to a new dataset.

        Args:
            data_in: Input passed to the transform to generate output dataset. Should support ``__getitem__`` and ``__len__`` operations. Can be a Deep Lake dataset.
            ds_out (Dataset, optional): The dataset object to which the transform will get written. If this is not provided, the ComputeFunction will operate in-place, which means that data will be written to tensors in ``data_in`` .
                All tensors modified in the ComputeFunction should already be defined in ``ds_out``. It's initial state should be either:
                - Empty i.e. all tensors have no samples. In this case all samples are added to the dataset.
                - All tensors are populated and have same length. In this case new samples are appended to the dataset.
            num_workers (int): The number of workers to use for performing the transform. Defaults to 0. When set to 0, it will always use serial processing, irrespective of the scheduler.
            scheduler (str): The scheduler to be used to compute the transformation. Supported values include: ``serial``, ``threaded``, and ``processed``.
                Defaults to 'threaded'.
            progressbar (bool): Displays a progress bar if ``True`` (default).
            skip_ok (bool): If ``True``, skips the check for output tensors generated. This allows the user to skip certain tensors in the function definition.
                This is especially useful for inplace transformations in which certain tensors are not modified. Defaults to ``False``.
            check_lengths (bool): If ``True``, checks whether ``ds_out`` has tensors of same lengths initially.
            pad_data_in (bool): If ``True``, pads tensors of ``data_in`` to match the length of the largest tensor in ``data_in``.
                Defaults to ``False``.
            read_only_ok (bool): If ``True`` and output dataset is same as input dataset, the read-only check is skipped.
                Defaults to False.
            cache_size (int): Cache size to be used by transform per worker.
            checkpoint_interval (int): If > 0, the ComputeFunction will be checkpointed with a commit every ``checkpoint_interval`` input samples to avoid restarting full transform due to intermitten failures. If the transform is interrupted, the intermediate data is deleted and the dataset is reset to the last commit.
                If <= 0, no checkpointing is done. Checkpoint interval should be a multiple of num_workers if ``num_workers`` > 0. Defaults to 0.
            ignore_errors (bool): If ``True``, input samples that causes transform to fail will be skipped and the errors will be ignored **if possible**.
            verbose (bool): If ``True``, prints additional information about the transform.
            **kwargs: Additional arguments.

        Raises:
            InvalidInputDataError: If ``data_in`` passed to transform is invalid. It should support ``__getitem__`` and ``__len__`` operations. Using scheduler other than ``threaded`` with deeplake dataset having base storage as memory as ``data_in`` will also raise this.
            InvalidOutputDatasetError: If all the tensors of ``ds_out`` passed to transform don't have the same length. Using scheduler other than "threaded" with deeplake dataset having base storage as memory as ``ds_out`` will also raise this.
            TensorMismatchError: If one or more of the outputs generated during transform contain different tensors than the ones present in 'ds_out' provided to transform.
            UnsupportedSchedulerError: If the scheduler passed is not recognized. Supported values include: ``serial``, ``threaded``, and ``processed``.
            TransformError: All other exceptions raised if there are problems while running the pipeline.
            ValueError: If ``num_workers`` > 0 and ``checkpoint_interval`` is not a multiple of ``num_workers`` or if ``checkpoint_interval`` > 0 and ds_out is None.
            AllSamplesSkippedError: If all samples are skipped during execution of the Pipeline.
            ModuleNotInstalledException: If the module ``ray`` is not installed and the scheduler is set to ``ray``.

        # noqa: DAR401

        Example:

            # Suppose we have a series of operations that we want to perform in parallel on images using reusable pipelines.
            # We use the pipeline to ingest the transfomed data from one dataset to another dataset.

            # First, we define the ComputeFunctions that will be used in the pipeline
            @deeplake.compute
            def flip_vertical(sample_in, sample_out):
                sample_out.append({'labels': sample_in.labels.numpy(),
                                    'images': np.flip(sample_in.images.numpy(), axis = 0)})

            @deeplake.compute
            def resize(sample_in, sample_out, new_size):
                sample_out.append({"labels": sample_in.labels.numpy(),
                                    "images": np.array(Image.fromarray(sample_in.images.numpy()).resize(new_size))})

            # Append the label and image to the output sample
            sample_out.labels.append(sample_in.labels.numpy())
            sample_out.images.append(np.array(Image.fromarray(sample_in.images.numpy()).resize(new_size)))

            # We can define the pipeline using:
            pipeline = deeplake.compose([flip_vertical(), resize(new_size = (64,64))])

            # Finally, we can evaluate the pipeline using:
            pipeline.eval(ds_in, ds_out, num_workers = 4)

        Note:
            ``pad_data_in`` is only applicable if ``data_in`` is a Deep Lake dataset.

        """
        num_workers, scheduler = sanitize_workers_scheduler(num_workers, scheduler)
        overwrite = ds_out is None
        deeplake_reporter.feature_report(
            feature_name="eval",
            parameters={"Num_Workers": str(num_workers), "Scheduler": scheduler},
        )
        check_transform_data_in(data_in, scheduler)

        data_in, original_data_in, initial_padding_state = prepare_data_in(
            data_in, pad_data_in, overwrite
        )
        target_ds = data_in if overwrite else ds_out

        check_transform_ds_out(
            target_ds, scheduler, check_lengths, read_only_ok and overwrite
        )

        # if overwrite then we've already flushed and autocheckecked out data_in which is target_ds now
        if not overwrite:
            target_ds.flush()
            auto_checkout(target_ds)

        compute_provider = get_compute_provider(scheduler, num_workers)
        compute_id = str(uuid4().hex)
        target_ds._send_compute_progress(compute_id=compute_id, start=True, progress=0)

        initial_autoflush = target_ds.storage.autoflush
        target_ds.storage.autoflush = False

        if not check_lengths or read_only_ok:
            skip_ok = True

        checkpointing_enabled = checkpoint_interval > 0
        total_samples = len_data_in(data_in)
        if checkpointing_enabled:
            check_checkpoint_interval(
                data_in,
                checkpoint_interval,
                num_workers,
                overwrite,
                verbose,
            )
            datas_in = [
                data_in[i : i + checkpoint_interval]
                for i in range(0, len_data_in(data_in), checkpoint_interval)
            ]

        else:
            datas_in = [data_in]

        samples_processed = 0
        desc = get_pbar_description(self.functions)
        if progressbar:
            pbar = get_progress_bar(len_data_in(data_in), desc)
            pqueue = compute_provider.create_queue()
        else:
            pbar, pqueue = None, None
        try:
            desc = desc.split()[1]
            completed = False
            progress = 0.0
            for data_in in datas_in:
                if checkpointing_enabled:
                    target_ds._commit(
                        f"Auto-commit during deeplake.compute of {desc} after {progress}% progress",
                        None,
                        False,
                        is_checkpoint=True,
                        total_samples_processed=samples_processed,
                    )
                progress = round(
                    (samples_processed + len_data_in(data_in)) / total_samples * 100, 2
                )
                end = progress == 100
                progress_args = {
                    "compute_id": compute_id,
                    "progress": progress,
                    "end": end,
                }

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
                        read_only_ok and overwrite,
                        cache_size,
                        pbar,
                        pqueue,
                        ignore_errors,
                        **kwargs,
                    )
                    target_ds._send_compute_progress(**progress_args, status="success")
                    samples_processed += len_data_in(data_in)
                    completed = end
                except Exception as e:
                    if checkpointing_enabled:
                        print(
                            "Transform failed. Resetting back to last committed checkpoint."
                        )
                        target_ds.reset(force=True)
                    target_ds._send_compute_progress(**progress_args, status="failed")
                    index, sample, suggest = None, None, False
                    if isinstance(e, TransformError):
                        index, sample, suggest = e.index, e.sample, e.suggest
                        if checkpointing_enabled and isinstance(index, int):
                            index = samples_processed + index
                        e = e.__cause__  # type: ignore
                    if isinstance(e, AllSamplesSkippedError):
                        raise e
                    raise TransformError(
                        index=index,
                        sample=sample,
                        samples_processed=samples_processed,
                        suggest=suggest,
                    ) from e
                finally:
                    reload_and_rechunk(
                        overwrite,
                        original_data_in,
                        target_ds,
                        initial_autoflush,
                        pad_data_in,
                        initial_padding_state,
                        kwargs,
                        completed,
                    )
        finally:
            close_states(compute_provider, pbar, pqueue)

    def run(
        self,
        data_in,
        target_ds: deeplake.Dataset,
        compute: ComputeProvider,
        num_workers: int,
        scheduler: str,
        progressbar: bool = True,
        overwrite: bool = False,
        skip_ok: bool = False,
        read_only: bool = False,
        cache_size: int = 16,
        pbar=None,
        pqueue=None,
        ignore_errors: bool = False,
        **kwargs,
    ):
        """Runs the pipeline on the input data to produce output samples and stores in the dataset.
        This receives arguments processed and sanitized by the Pipeline.eval method.
        """
        if isinstance(data_in, deeplake.Dataset):
            dataset_read(data_in)
        slices, offsets = create_slices(data_in, num_workers)
        storage = get_base_storage(target_ds.storage)
        class_label_tensors = (
            [
                tensor.key
                for tensor in target_ds.tensors.values()
                if tensor.base_htype == "class_label"
                and not read_only
                and not tensor.meta._disable_temp_transform
            ]
            if not kwargs.get("disable_label_sync")
            else []
        )
        label_temp_tensors = {}

        visible_tensors = list(target_ds.tensors)
        visible_tensors = [target_ds[t].key for t in visible_tensors]

        if not read_only:
            for tensor in class_label_tensors:
                actual_tensor = target_ds[tensor]
                temp_tensor = (
                    f"__temp{relpath(tensor, target_ds.group_index)}_{uuid4().hex[:4]}"
                )
                with target_ds:
                    temp_tensor_obj = target_ds.create_tensor(
                        temp_tensor,
                        htype="class_label",
                        dtype=actual_tensor.dtype,
                        hidden=True,
                        create_sample_info_tensor=False,
                        create_shape_tensor=False,
                        create_id_tensor=False,
                    )
                    temp_tensor_obj.meta._disable_temp_transform = True
                    label_temp_tensors[tensor] = temp_tensor_obj.key
                target_ds.flush()

        tensors = list(target_ds._tensors(include_disabled=False))
        tensors = [target_ds[t].key for t in tensors]
        tensors = list(set(tensors) - set(class_label_tensors))

        group_index = target_ds.group_index
        version_state = target_ds.version_state
        if isinstance(storage, MemoryProvider):
            storages = [storage] * len(slices)
        else:
            storages = [storage.copy() for _ in slices]
        extend_only = kwargs.get("extend_only")
        args = (
            group_index,
            tensors,
            visible_tensors,
            label_temp_tensors,
            self,
            version_state,
            target_ds.link_creds,
            skip_ok,
            extend_only,
            cache_size,
            ignore_errors,
        )
        map_inp = zip(slices, offsets, storages, repeat(args))
        try:
            if progressbar:
                desc = get_pbar_description(self.functions)
                result = compute.map_with_progress_bar(
                    store_data_slice_with_pbar,
                    map_inp,
                    total_length=len_data_in(data_in),
                    desc=desc,
                    pbar=pbar,
                    pqueue=pqueue,
                )
            else:
                result = compute.map(store_data_slice, map_inp)
        except Exception:
            for tensor in label_temp_tensors.values():
                target_ds.delete_tensor(tensor)
            raise

        if read_only:
            return

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

        if ignore_errors:
            transform_summary(data_in, result)

        for res in result["error"]:
            if res is not None:
                print(res["traceback"])
                print(
                    "The above exception was the direct cause of the following exception:\n"
                )
                raise res["raise"]


def compose(functions: List[ComputeFunction]):  # noqa: DAR101, DAR102, DAR201, DAR401
    """Takes a list of functions decorated using :func:`deeplake.compute` and creates a pipeline that can be evaluated using .eval

    Example::

        pipeline = deeplake.compose([my_fn(a=3), another_function(b=2)])
        pipeline.eval(data_in, ds_out, scheduler="processed", num_workers=2)

    The ``eval`` method evaluates the pipeline/transform function.

    It has the following arguments:

    - ``data_in``: Input passed to the transform to generate output dataset.

        - It should support ``__getitem__`` and ``__len__``. This can be a Deep Lake dataset.

    - ``ds_out (Dataset, optional)``: The dataset object to which the transform will get written.

        - If this is not provided, data_in will be overwritten if it is a Deep Lake dataset, otherwise error will be raised.
        - It should have all keys being generated in output already present as tensors.
        - It's initial state should be either:

            - Empty i.e. all tensors have no samples. In this case all samples are added to the dataset.
            - All tensors are populated and have same length. In this case new samples are appended to the dataset.

    - ``num_workers (int)``: The number of workers to use for performing the transform.

        - Defaults to 0. When set to 0, it will always use serial processing, irrespective of the scheduler.

    - ``scheduler (str)``: The scheduler to be used to compute the transformation.

        - Supported values include: 'serial', 'threaded', and 'processed'. Defaults to 'threaded'.

    - ``progressbar (bool)``: Displays a progress bar if True (default).

    - ``skip_ok (bool)``: If True, skips the check for output tensors generated.

        - This allows the user to skip certain tensors in the function definition.
        - This is especially useful for inplace transformations in which certain tensors are not modified. Defaults to ``False``.

    - ``ignore_errors (bool)``: If ``True``, input samples that causes transform to fail will be skipped and the errors will be ignored **if possible**.

    It raises the following errors:

    - ``InvalidInputDataError``: If data_in passed to transform is invalid. It should support ``__getitem__`` and ``__len__`` operations. Using scheduler other than "threaded" with deeplake dataset having base storage as memory as data_in will also raise this.

    - ``InvalidOutputDatasetError``: If all the tensors of ds_out passed to transform don't have the same length. Using scheduler other than "threaded" with deeplake dataset having base storage as memory as ds_out will also raise this.

    - ``TensorMismatchError``: If one or more of the outputs generated during transform contain different tensors than the ones present in 'ds_out' provided to transform.

    - ``UnsupportedSchedulerError``: If the scheduler passed is not recognized. Supported values include: 'serial', 'threaded', and 'processed'.

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

    The output should be appended/extended to the second argument in a deeplake like syntax.

    Any value returned by the fn will be ignored.

    Example::

        @deeplake.compute
        def my_fn(sample_in: Any, samples_out, my_arg0, my_arg1=0):
            samples_out.my_tensor.append(my_arg0 * my_arg1)

        # This transform can be used using the eval method in one of these 2 ways:-

        # Directly evaluating the method
        # here arg0 and arg1 correspond to the 3rd and 4th argument in my_fn
        my_fn(arg0, arg1).eval(data_in, ds_out, scheduler="threaded", num_workers=5)

        # As a part of a Transform pipeline containing other functions
        pipeline = deeplake.compose([my_fn(a, b), another_function(x=2)])
        pipeline.eval(data_in, ds_out, scheduler="processed", num_workers=2)

    The ``eval`` method evaluates the pipeline/transform function.

    It has the following arguments:

    - ``data_in``: Input passed to the transform to generate output dataset.

        - It should support ``__getitem__`` and ``__len__``. This can be a Deep Lake dataset.

    - ``ds_out (Dataset, optional)``: The dataset object to which the transform will get written.

        - If this is not provided, data_in will be overwritten if it is a Deep Lake dataset, otherwise error will be raised.
        - It should have all keys being generated in output already present as tensors.
        - It's initial state should be either:

            - Empty i.e. all tensors have no samples. In this case all samples are added to the dataset.
            - All tensors are populated and have same length. In this case new samples are appended to the dataset.

    - ``num_workers (int)``: The number of workers to use for performing the transform.

        - Defaults to 0. When set to 0, it will always use serial processing, irrespective of the scheduler.

    - ``scheduler (str)``: The scheduler to be used to compute the transformation.

        - Supported values include: 'serial', 'threaded', and 'processed'. Defaults to 'threaded'.

    - ``progressbar (bool)``: Displays a progress bar if ``True`` (default).

    - ``skip_ok (bool)``: If ``True``, skips the check for output tensors generated.

        - This allows the user to skip certain tensors in the function definition.
        - This is especially useful for inplace transformations in which certain tensors are not modified. Defaults to ``False``.

    - ``check_lengths (bool)``: If ``True``, checks whether ``ds_out`` has tensors of same lengths initially.

    - ``pad_data_in (bool)``: If ``True``, pads tensors of ``data_in`` to match the length of the largest tensor in ``data_in``. Defaults to ``False``.

    - ``ignore_errors (bool)``: If ``True``, input samples that causes transform to fail will be skipped and the errors will be ignored **if possible**.

    Note:
        ``pad_data_in`` is only applicable if ``data_in`` is a Deep Lake dataset.

    It raises the following errors:

    - ``InvalidInputDataError``: If ``data_in`` passed to transform is invalid. It should support ``__getitem__`` and ``__len__`` operations. Using scheduler other than "threaded" with deeplake dataset having base storage as memory as ``data_in`` will also raise this.

    - ``InvalidOutputDatasetError``: If all the tensors of ``ds_out`` passed to transform don't have the same length. Using scheduler other than "threaded" with deeplake dataset having base storage as memory as ``ds_out`` will also raise this.

    - ``TensorMismatchError``: If one or more of the outputs generated during transform contain different tensors than the ones present in ``ds_out`` provided to transform.

    - ``UnsupportedSchedulerError``: If the scheduler passed is not recognized. Supported values include: 'serial', 'threaded', and 'processed'.

    - ``TransformError``: All other exceptions raised if there are problems while running the pipeline.
    """

    # Note: the name and signature of this method is checked for in deeplake/core/query/filter.py:filter_dataset()
    def inner(*args, **kwargs):
        return ComputeFunction(fn, args, kwargs, name)

    return inner
