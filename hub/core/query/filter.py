from typing import Callable, List, Optional, Sequence
from uuid import uuid4

import hub

from hub.core.io import SampleStreaming
from hub.util.compute import get_compute_provider
from hub.util.dataset import map_tensor_keys
from time import time

import inspect
import threading
from queue import Queue
from collections import defaultdict

from hub.util.exceptions import FilterError
from hub.util.hash import hash_inputs


# Frequency for sending progress events and writing to vds
_UPDATE_FREQUENCY = 5  # seconds


_LAST_UPDATED_TIMES = defaultdict(time)


def _counter(id):
    """A method which returns True only every `_UPDATE_FREQUENCY` seconds for each id.
    Used for sending query progress update events and writing to vds.
    """
    last_updated_time = _LAST_UPDATED_TIMES[id]
    curr_time = time()
    if curr_time - last_updated_time > _UPDATE_FREQUENCY:
        _LAST_UPDATED_TIMES[id] = curr_time
        return True
    return False


def _del_counter(id):
    _LAST_UPDATED_TIMES.pop(id, None)


def _filter_function_to_query_text(filter_function):
    if isinstance(filter_function, hub.core.query.DatasetQuery):
        query_text = filter_function._query
    else:
        try:
            query_text = inspect.getsource(filter_function)
        except OSError:
            query_text = (
                "UDF: "
                + getattr(
                    filter_function, "__name__", filter_function.__class__.__name__
                )
                + "@"
                + str(uuid4().hex)
            )  # because function name alone is not unique enough
    return query_text


def filter_dataset(
    dataset: hub.Dataset,
    filter_function: Callable[[hub.Dataset], bool],
    num_workers: int = 0,
    scheduler: str = "threaded",
    progressbar: bool = True,
    store_result: bool = False,
    result_path: Optional[str] = None,
    result_ds_args: Optional[dict] = None,
) -> hub.Dataset:
    index_map: List[int]

    tm = time()

    query_text = _filter_function_to_query_text(filter_function)
    vds = (
        dataset._get_empty_vds(result_path, query=query_text, **(result_ds_args or {}))
        if store_result
        else None
    )

    index_map = None
    try:
        if num_workers > 0:
            index_map = filter_with_compute(
                dataset,
                filter_function,
                num_workers,
                scheduler,
                progressbar,
                query_text,
                vds,
            )
        else:
            index_map = filter_inplace(
                dataset,
                filter_function,
                progressbar,
                query_text,
                vds,
            )
    except Exception as e:
        if vds:
            vds.info["error"] = str(e)
            raise (e)

    ds = dataset[index_map]
    ds._is_filtered_view = True

    ds._query = query_text
    ds._source_ds_idx = dataset.index.to_json()
    ds._created_at = tm
    if vds:
        ds._vds = vds
    return ds  # type: ignore [this is fine]


def _get_vds_thread(vds: hub.Dataset, queue: Queue, num_samples: int):
    """Creates a thread which writes to a vds in background.

    Args:
        vds: (hub.Dataset) The vds to write to.
        queue: (Queue) Queue to pop progress info from.
            Each item in the queue should be of form Tuple[int, bool],
            where the int is a sample index and the bool is whether
            or not to include the sample index in the vds.
        num_samples (int): Total number of samples in the source dataset.
    """
    id = str(uuid4().hex)

    def loop():
        processed = 0
        while True:
            index, include = queue.get()
            vds.info["samples_processed"] += 1
            if include:
                vds.VDS_INDEX.append(index)
            processed += 1
            if processed == num_samples:
                vds.flush()
                _del_counter(id)
                break
            if _counter(id):
                vds.flush()

    return threading.Thread(target=loop)


def filter_with_compute(
    dataset: hub.Dataset,
    filter_function: Callable,
    num_workers: int,
    scheduler: str,
    progressbar: bool = True,
    query_text: Optional[str] = None,
    vds: Optional[hub.Dataset] = None,
) -> List[int]:

    blocks = SampleStreaming(dataset, tensors=map_tensor_keys(dataset)).list_blocks()
    compute = get_compute_provider(scheduler=scheduler, num_workers=num_workers)

    num_samples = len(dataset)

    if vds:
        vds.autoflush = False
        vds.info["total_samples"] = num_samples
        vds.info["samples_processed"] = 0
        vds_queue = compute.create_queue()
        vds_thread = _get_vds_thread(vds, vds_queue, num_samples)
        vds_thread.start()

    query_id = hash_inputs(dataset.path, dataset.pending_commit_id, query_text)

    progress = {"value": 0}
    # Callback for sending query progress
    def _event_callback():
        progress["value"] += 1
        if _counter(query_id):
            dataset._send_query_progress(
                query_text=query_text,
                query_id=query_id,
                progress=int(100 * progress["value"] / num_samples),
            )

    def filter_slice(indices: Sequence[int]):
        result = list()
        for i in indices:
            if filter_function(dataset[i]):
                result.append(i)
                if vds:
                    vds_queue.put((i, True))
            elif vds:
                vds_queue.put((i, False))
            _event_callback()
        return result

    def pg_filter_slice(pg_callback, indices: Sequence[int]):
        result = list()
        for i in indices:
            if filter_function(dataset[i]):
                result.append(i)
                if vds:
                    vds_queue.put((i, True))
            elif vds:
                vds_queue.put((i, False))
            pg_callback(1)
            _event_callback()
        return result

    result: Sequence[List[int]]
    idx: List[List[int]] = [block.indices() for block in blocks]

    dataset._send_query_progress(
        query_text=query_text, query_id=query_id, start=True, progress=0
    )

    try:
        if progressbar:
            result = compute.map_with_progressbar(pg_filter_slice, idx, total_length=len(dataset))  # type: ignore
        else:
            result = compute.map(filter_slice, idx)  # type: ignore
        index_map = [k for x in result for k in x]  # unfold the result map
        dataset._send_query_progress(
            query_text=query_text,
            query_id=query_id,
            end=True,
            progress=100,
            status="success",
        )
    except Exception as e:
        dataset._send_query_progress(
            query_text=query_text,
            query_id=query_id,
            end=True,
            progress=100,
            status="failed",
        )
        raise FilterError(e)

    finally:
        compute.close()
        if vds:
            vds.autoflush = True
            vds_thread.join()
            if hasattr(vds_queue, "close"):
                vds_queue.close()
        _del_counter(query_id)
    return index_map


def filter_inplace(
    dataset: hub.Dataset,
    filter_function: Callable,
    progressbar: bool,
    query_text: Optional[str] = None,
    vds: Optional[hub.Dataset] = None,
) -> List[int]:
    index_map: List[int] = list()

    it = enumerate(dataset)
    num_samples = len(dataset)
    if vds:
        vds.autoflush = False
        vds.info["total_samples"] = len(dataset)
        vds.info["samples_processed"] = 0
        vds_queue = Queue()
        vds_thread = _get_vds_thread(vds, vds_queue, num_samples)
        vds_thread.start()
    if progressbar:
        from tqdm import tqdm  # type: ignore

        it = tqdm(it, total=num_samples)

    query_id = hash_inputs(dataset.path, dataset.pending_commit_id, query_text)

    dataset._send_query_progress(
        query_text=query_text, query_id=query_id, start=True, progress=0
    )

    try:
        for i, sample_in in it:
            if filter_function(sample_in):
                index_map.append(i)
                if vds:
                    vds_queue.put((i, True))
            elif vds:
                vds_queue.put((i, False))
            if _counter(query_id):
                dataset._send_query_progress(
                    query_text=query_text,
                    query_id=query_id,
                    progress=int(i * 100 / num_samples),
                    status="success",
                )
        dataset._send_query_progress(
            query_text=query_text,
            query_id=query_id,
            end=True,
            progress=100,
            status="success",
        )
    except Exception as e:
        dataset._send_query_progress(
            query_text=query_text,
            query_id=query_id,
            end=True,
            progress=100,
            status="failed",
        )
        raise (e)
    finally:
        if vds:
            vds.autoflush = True
            vds_thread.join()
        _del_counter(query_id)

    return index_map
