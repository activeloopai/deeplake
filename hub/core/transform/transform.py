import hub
import math
from typing import List, Callable, Optional
from itertools import repeat
from hub.core.compute.provider import ComputeProvider
from hub.core.compute.thread import ThreadProvider
from hub.core.compute.process import ProcessProvider
from hub.core.compute.serial import SerialProvider
from hub.core.ipc import Server
from hub.util.bugout_reporter import hub_reporter
from hub.util.compute import get_compute_provider
from hub.util.remove_cache import get_base_storage, get_dataset_with_zero_size_cache
from hub.util.transform import (
    check_transform_data_in,
    check_transform_ds_out,
    get_pbar_description,
    store_data_slice,
)
from hub.util.encoder import merge_all_chunk_id_encoders, merge_all_tensor_metas
from hub.util.exceptions import (
    HubComposeEmptyListError,
    HubComposeIncompatibleFunction,
    TransformError,
)

from tqdm import tqdm  # type: ignore
import time
import threading
import sys


class TransformFunction:
    def __init__(self, func, args, kwargs):
        """Creates a TransformFunction object that can be evaluated using .eval or used as a part of a Pipeline."""
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def eval(
        self,
        data_in,
        ds_out: hub.Dataset,
        num_workers: int = 0,
        scheduler: str = "threaded",
        progressbar: bool = True,
    ):
        """Evaluates the TransformFunction on data_in to produce an output dataset ds_out.

        Args:
            data_in: Input passed to the transform to generate output dataset. Should support __getitem__ and __len__. Can be a Hub dataset.
            ds_out (Dataset): The dataset object to which the transform will get written.
                Should have all keys being generated in output already present as tensors. It's initial state should be either:-
                - Empty i.e. all tensors have no samples. In this case all samples are added to the dataset.
                - All tensors are populated and have sampe length. In this case new samples are appended to the dataset.
            num_workers (int): The number of workers to use for performing the transform. Defaults to 0. When set to 0, it will always use serial processing, irrespective of the scheduler.
            scheduler (str): The scheduler to be used to compute the transformation. Supported values include: "serial", 'threaded', 'processed' and 'ray.
            progressbar (bool): Displays a progress bar if True (default).


        Raises:
            InvalidInputDataError: If data_in passed to transform is invalid. It should support __getitem__ and __len__ operations. Using scheduler other than "threaded" with hub dataset having base storage as memory as data_in will also raise this.
            InvalidOutputDatasetError: If all the tensors of ds_out passed to transform don't have the same length. Using scheduler other than "threaded" with hub dataset having base storage as memory as ds_out will also raise this.
            TensorMismatchError: If one or more of the outputs generated during transform contain different tensors than the ones present in 'ds_out' provided to transform.
            UnsupportedSchedulerError: If the scheduler passed is not recognized. Supported values include: "serial", 'threaded', 'processed' and 'ray'.
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
        ds_out: hub.Dataset,
        num_workers: int = 0,
        scheduler: str = "threaded",
        progressbar: bool = True,
    ):
        """Evaluates the pipeline on data_in to produce an output dataset ds_out.

        Args:
            data_in: Input passed to the transform to generate output dataset. Should support __getitem__ and __len__. Can be a Hub dataset.
            ds_out (Dataset): The dataset object to which the transform will get written.
                Should have all keys being generated in output already present as tensors. It's initial state should be either:-
                - Empty i.e. all tensors have no samples. In this case all samples are added to the dataset.
                - All tensors are populated and have sampe length. In this case new samples are appended to the dataset.
            num_workers (int): The number of workers to use for performing the transform. Defaults to 0. When set to 0, it will always use serial processing, irrespective of the scheduler.
            scheduler (str): The scheduler to be used to compute the transformation. Supported values include: "serial", 'threaded', 'processed' and 'ray'.
            progressbar (bool): Displays a progress bar if True (default).

        Raises:
            InvalidInputDataError: If data_in passed to transform is invalid. It should support __getitem__ and __len__ operations. Using scheduler other than "threaded" with hub dataset having base storage as memory as data_in will also raise this.
            InvalidOutputDatasetError: If all the tensors of ds_out passed to transform don't have the same length. Using scheduler other than "threaded" with hub dataset having base storage as memory as ds_out will also raise this.
            TensorMismatchError: If one or more of the outputs generated during transform contain different tensors than the ones present in 'ds_out' provided to transform.
            UnsupportedSchedulerError: If the scheduler passed is not recognized. Supported values include: "serial", 'threaded', 'processed' and 'ray'.
            TransformError: All other exceptions raised if there are problems while running the pipeline.
        """
        num_workers = max(num_workers, 0)
        if num_workers == 0:
            scheduler = "serial"

        if isinstance(data_in, hub.Dataset):
            data_in = get_dataset_with_zero_size_cache(data_in)

        hub_reporter.feature_report(
            feature_name="eval",
            parameters={"Num_Workers": str(num_workers), "Scheduler": scheduler},
        )

        check_transform_data_in(data_in, scheduler)
        check_transform_ds_out(ds_out, scheduler)

        ds_out.flush()
        initial_autoflush = ds_out.storage.autoflush
        ds_out.storage.autoflush = False

        tensors = list(ds_out.tensors)

        compute_provider = get_compute_provider(scheduler, num_workers)

        try:
            self.run(
                data_in, ds_out, tensors, compute_provider, num_workers, progressbar
            )
        except Exception as e:
            raise TransformError(e)
        finally:
            compute_provider.close()

        ds_out.storage.autoflush = initial_autoflush

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
        ds_out: hub.Dataset,
        tensors: List[str],
        compute: ComputeProvider,
        num_workers: int,
        progressbar: bool = True,
    ):
        """Runs the pipeline on the input data to produce output samples and stores in the dataset.
        This receives arguments processed and sanitized by the Pipeline.eval method.
        """
        is_serial = isinstance(compute, SerialProvider)
        num_workers = max(num_workers, 1)
        size = math.ceil(len(data_in) / num_workers)
        slices = [data_in[i * size : (i + 1) * size] for i in range(num_workers)]

        output_base_storage = get_base_storage(ds_out.storage)
        version_state = ds_out.version_state
        tensors = [ds_out.tensors[t].key for t in tensors]

        ret = {}

        def _run(progress_port=None):
            ret["metas_and_encoders"] = compute.map(
                store_data_slice,
                zip(
                    slices,
                    repeat((output_base_storage, ds_out.group_index)),  # type: ignore
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

        metas_and_encoders = ret["metas_and_encoders"]

        all_tensor_metas, all_chunk_id_encoders = zip(*metas_and_encoders)
        merge_all_tensor_metas(all_tensor_metas, ds_out)
        merge_all_chunk_id_encoders(all_chunk_id_encoders, ds_out)


def compose(functions: List[TransformFunction]):
    """Takes a list of functions decorated using hub.compute and creates a pipeline that can be evaluated using .eval"""
    if not functions:
        raise HubComposeEmptyListError
    for index, fn in enumerate(functions):
        if not isinstance(fn, TransformFunction):
            raise HubComposeIncompatibleFunction(index)
    return Pipeline(functions)


def compute(fn):
    """Compute is a decorator for functions.
    The functions should have atleast 2 argument, the first two will correspond to sample_in and samples_out.
    There can be as many other arguments as required.
    The output should be appended/extended to the second argument in a hub like syntax.
    Any value returned by the fn will be ignored.

    Example::

        @hub.compute
        def your_function(sample_in: Any, samples_out, your_arg0, your_arg1=0):
            samples_out.your_tensor.append(your_arg0 * your_arg1)
    """

    def inner(*args, **kwargs):
        return TransformFunction(fn, args, kwargs)

    return inner
