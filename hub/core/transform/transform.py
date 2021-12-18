import hub
import math
from typing import List, Callable, Optional
from itertools import repeat
from hub.constants import FIRST_COMMIT_ID
from hub.core.compute.provider import ComputeProvider
from hub.core.ipc import Server
from hub.util.bugout_reporter import hub_reporter
from hub.util.chunk_paths import get_chunk_paths
from hub.util.compute import get_compute_provider
from hub.util.remove_cache import get_base_storage, get_dataset_with_zero_size_cache
from hub.util.transform import (
    check_transform_data_in,
    check_transform_ds_out,
    get_pbar_description,
    store_data_slice,
)
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

from tqdm import tqdm  # type: ignore
import time
import threading
import sys

from hub.util.version_control import auto_checkout, load_meta


class TransformFunction:
    def __init__(self, func, args, kwargs):
        """Creates a TransformFunction object that can be evaluated using .eval or used as a part of a Pipeline."""
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
    ):
        """Evaluates the TransformFunction on data_in to produce an output dataset ds_out.

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


        Raises:
            InvalidInputDataError: If data_in passed to transform is invalid. It should support \__getitem__ and \__len__ operations. Using scheduler other than "threaded" with hub dataset having base storage as memory as data_in will also raise this.
            InvalidOutputDatasetError: If all the tensors of ds_out passed to transform don't have the same length. Using scheduler other than "threaded" with hub dataset having base storage as memory as ds_out will also raise this.
            TensorMismatchError: If one or more of the outputs generated during transform contain different tensors than the ones present in 'ds_out' provided to transform.
            UnsupportedSchedulerError: If the scheduler passed is not recognized. Supported values include: 'serial', 'threaded', 'processed' and 'ray'.
        """

        pipeline = Pipeline([self])
        pipeline.eval(data_in, ds_out, num_workers, scheduler, progressbar)


class Pipeline:
    def __init__(self, functions: List[TransformFunction]):
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
        target_ds.flush()
        # if not the head node, checkout to an auto branch that is newly created
        auto_checkout(target_ds.version_state, target_ds.storage)

        initial_autoflush = target_ds.storage.autoflush
        target_ds.storage.autoflush = False

        overwrite = ds_out is None
        if overwrite:
            original_data_in.clear_cache()

        compute_provider = get_compute_provider(scheduler, num_workers)
        try:
            self.run(
                data_in,
                target_ds,
                compute_provider,
                num_workers,
                progressbar,
                overwrite,
            )
        except Exception as e:
            raise TransformError(e)
        finally:
            compute_provider.close()
            target_ds.storage.autoflush = initial_autoflush

    def _run_with_progbar(
        self, func: Callable, ret: dict, total: int, desc: Optional[str] = ""
    ):
        """
        Args:
            func (Callable): Function to be executed
            ret (dict): `func` should place its return value in this dictionary
            total (int): Total number of steps in the progress bar
            desc (str, Optional): Description for the progress bar

        Raises:
            Exception: If any worker encounters an error, it is raised
        """
        ismac = sys.platform == "darwin"
        progress = {"value": 0, "error": None}

        def callback(data):
            if isinstance(data, str):
                progress["error"] = data
            else:
                progress["value"] += data

        server = Server(callback)
        port = server.port
        thread = threading.Thread(target=func, args=(port,), daemon=ismac)
        thread.start()
        try:
            for i in tqdm(range(total), desc=desc):
                while i + 1 > progress["value"]:
                    time.sleep(1)
                    err = progress["error"]
                    if err:
                        raise Exception(err)
        finally:
            if ismac:
                while not ret:  # thread.join() takes forever on mac
                    time.sleep(1)
            else:
                thread.join()
            server.stop()

    def run(
        self,
        data_in,
        target_ds: hub.Dataset,
        compute: ComputeProvider,
        num_workers: int,
        progressbar: bool = True,
        overwrite: bool = False,
    ):
        """Runs the pipeline on the input data to produce output samples and stores in the dataset.
        This receives arguments processed and sanitized by the Pipeline.eval method.
        """
        size = math.ceil(len(data_in) / num_workers)
        slices = [data_in[i * size : (i + 1) * size] for i in range(num_workers)]
        storage = get_base_storage(target_ds.storage)
        group_index = target_ds.group_index  # type: ignore
        version_state = target_ds.version_state

        tensors = list(target_ds.tensors)
        tensors = [target_ds.tensors[t].key for t in tensors]

        ret = {}

        def _run(progress_port=None):
            ret["metas_and_encoders"] = compute.map(
                store_data_slice,
                zip(
                    slices,
                    repeat((storage, group_index)),  # type: ignore
                    repeat(tensors),
                    repeat(self),
                    repeat(version_state),
                    repeat(progress_port),
                ),
            )

        if progressbar:
            self._run_with_progbar(
                _run, ret, len(data_in), get_pbar_description(self.functions)
            )
        else:
            _run()

        if overwrite:
            chunk_paths = get_chunk_paths(target_ds, tensors)
            # TODO:
            # delete_chunks(chunk_paths, storage, compute)

        metas_and_encoders = ret["metas_and_encoders"]

        (
            all_tensor_metas,
            all_chunk_id_encoders,
            all_tile_encoders,
            all_chunk_commit_sets,
            all_commit_diffs,
        ) = zip(*metas_and_encoders)
        all_num_samples = []
        for tensor_meta_dict in all_tensor_metas:
            num_samples_dict = {k: v.length for k, v in tensor_meta_dict.items()}
            all_num_samples.append(num_samples_dict)
        merge_all_commit_diffs(all_commit_diffs, target_ds, storage, overwrite)
        merge_all_tile_encoders(
            all_tile_encoders, all_num_samples, target_ds, storage, overwrite
        )
        merge_all_tensor_metas(all_tensor_metas, target_ds, storage, overwrite)
        merge_all_chunk_id_encoders(
            all_chunk_id_encoders, target_ds, storage, overwrite
        )
        if target_ds.commit_id is not None:
            merge_all_commit_chunk_sets(
                all_chunk_commit_sets, target_ds, storage, overwrite
            )


def compose(functions: List[TransformFunction]):  # noqa: DAR101, DAR102, DAR201, DAR401
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
        if not isinstance(fn, TransformFunction):
            raise HubComposeIncompatibleFunction(index)
    return Pipeline(functions)


def compute(fn):  # noqa: DAR101, DAR102, DAR201, DAR401
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

    It raises the following errors:-

    - InvalidInputDataError: If data_in passed to transform is invalid. It should support \__getitem__ and \__len__ operations. Using scheduler other than "threaded" with hub dataset having base storage as memory as data_in will also raise this.
    - InvalidOutputDatasetError: If all the tensors of ds_out passed to transform don't have the same length. Using scheduler other than "threaded" with hub dataset having base storage as memory as ds_out will also raise this.
    - TensorMismatchError: If one or more of the outputs generated during transform contain different tensors than the ones present in 'ds_out' provided to transform.
    - UnsupportedSchedulerError: If the scheduler passed is not recognized. Supported values include: 'serial', 'threaded', 'processed' and 'ray'.
    - TransformError: All other exceptions raised if there are problems while running the pipeline.
    """

    def inner(*args, **kwargs):
        return TransformFunction(fn, args, kwargs)

    return inner
