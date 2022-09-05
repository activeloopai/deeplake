from typing import Callable, List, Optional, Sequence, Dict
from uuid import uuid4

import hub

from hub.core.io import SampleStreaming
from hub.core.query.query import DatasetQuery
from hub.util.compute import get_compute_provider
from hub.util.dataset import map_tensor_keys
from hub.constants import QUERY_PROGRESS_UPDATE_FREQUENCY
from time import time

import inspect
import threading
from queue import Queue
from collections import defaultdict

from hub.util.exceptions import FilterError
from hub.util.hash import hash_inputs


_LAST_UPDATED_TIMES: Dict = defaultdict(time)


def _counter(id):
    """A method which returns True only every `QUERY_PROGRESS_UPDATE_FREQUENCY` seconds for each id.
    Used for sending query progress update events and writing to vds.
    """
    last_updated_time = _LAST_UPDATED_TIMES[id]
    curr_time = time()
    if curr_time - last_updated_time > QUERY_PROGRESS_UPDATE_FREQUENCY:
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
        except (OSError, TypeError):
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
    save_result: bool = False,
    result_path: Optional[str] = None,
    result_ds_args: Optional[dict] = None,
) -> hub.Dataset:
    index_map: List[int]

    tm = time()

    query_text = _filter_function_to_query_text(filter_function)
    vds = (
        dataset._get_empty_vds(result_path, query=query_text, **(result_ds_args or {}))
        if save_result
        else None
    )

    index_map = None  # type: ignore
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
        raise e

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

    Returns:
        threading.Thread object
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
                    _event_callback()
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
                    _event_callback()
            elif vds:
                vds_queue.put((i, False))
                _event_callback()
            pg_callback(1)
        return result

    result: Sequence[List[int]]
    idx: List[List[int]] = [block.indices() for block in blocks]
    if vds:
        dataset._send_query_progress(
            query_text=query_text, query_id=query_id, start=True, progress=0
        )

    try:
        if progressbar:
            result = compute.map_with_progressbar(pg_filter_slice, idx, total_length=len(dataset))  # type: ignore
        else:
            result = compute.map(filter_slice, idx)  # type: ignore
        index_map = [k for x in result for k in x]  # unfold the result map
        if vds:
            dataset._send_query_progress(
                query_text=query_text,
                query_id=query_id,
                end=True,
                progress=100,
                status="success",
            )
    except Exception as e:
        if vds:
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
            if hasattr(vds_queue, "close"):
                vds_queue.close()
        _del_counter(query_id)
    if vds:
        vds.autoflush = True
        vds_thread.join()
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
        vds_queue: Queue = Queue()
        vds_thread = _get_vds_thread(vds, vds_queue, num_samples)
        vds_thread.start()
    if progressbar:
        from tqdm import tqdm  # type: ignore

        it = tqdm(it, total=num_samples)

    query_id = hash_inputs(dataset.path, dataset.pending_commit_id, query_text)

    if vds:
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
            if vds and _counter(query_id):
                dataset._send_query_progress(
                    query_text=query_text,
                    query_id=query_id,
                    progress=int(i * 100 / num_samples),
                    status="success",
                )
        if vds:
            dataset._send_query_progress(
                query_text=query_text,
                query_id=query_id,
                end=True,
                progress=100,
                status="success",
            )
    except Exception as e:
        if vds:
            dataset._send_query_progress(
                query_text=query_text,
                query_id=query_id,
                end=True,
                progress=100,
                status="failed",
            )
        raise (e)
    finally:
        _del_counter(query_id)

    if vds:
        vds.autoflush = True
        vds_thread.join()
    return index_map


def query_dataset(
    dataset: hub.Dataset,
    query: str,
    num_workers: int = 0,
    scheduler: str = "threaded",
    progressbar: bool = True,
    save_result: bool = False,
    result_path: Optional[str] = None,
    result_ds_args: Optional[Dict] = None,
) -> hub.Dataset:
    index_map: List[int]

    vds = (
        dataset._get_empty_vds(result_path, query=query, **(result_ds_args or {}))
        if save_result
        else None
    )
    index_map = query_inplace(dataset, query, progressbar, num_workers, scheduler, vds)
    ret = dataset[index_map]  # type: ignore [this is fine]
    ret._query = query
    if vds:
        ret._vds = vds
    return ret


def query_inplace(
    dataset: hub.Dataset,
    query: str,
    progressbar: bool,
    num_workers: int,
    scheduler: str,
    vds: Optional[hub.Dataset] = None,
) -> List[int]:

    num_samples = len(dataset)
    compute = (
        get_compute_provider(scheduler=scheduler, num_workers=num_workers)
        if num_workers > 0
        else None
    )
    query_id = hash_inputs(dataset.path, dataset.pending_commit_id, query)

    if vds:
        vds.autoflush = False
        vds.info["total_samples"] = num_samples
        vds.info["samples_processed"] = 0
        vds_queue = Queue() if num_workers == 0 else compute.create_queue()  # type: ignore
        vds_thread = _get_vds_thread(vds, vds_queue, num_samples)
        vds_thread.start()
        dataset._send_query_progress(
            query_text=query, query_id=query_id, start=True, progress=0
        )

    num_processed = {"value": 0}

    def update_vds(idx, include):
        if vds:
            vds_queue.put((idx, include))
            num_processed["value"] += 1
            if _counter(query_id):
                dataset._send_query_progress(
                    query_text=query,
                    query_id=query_id,
                    progress=int(num_processed["value"] * 100 / num_samples),
                    status="success",
                )

    class QuerySlice:
        def __init__(self, offset, size, dataset, query) -> None:
            self.offset = offset
            self.size = size
            self.dataset = dataset
            self.query = query

        def slice_dataset(self):
            return self.dataset[self.offset : (self.offset + self.size)]

    def subquery(query_slice: QuerySlice):
        dataset = query_slice.slice_dataset()
        query = query_slice.query

        if progressbar:
            from tqdm import tqdm

            bar = tqdm(total=len(dataset))

            def update(idx, include):
                bar.update(1)
                update_vds(idx, include)

            try:
                ds_query = DatasetQuery(dataset, query, update)
                ret = ds_query.execute()
            finally:
                bar.close()
        else:
            ret = DatasetQuery(dataset, query, update_vds).execute()
        return ret

    def pg_subquery(pg_callback, query_slice):
        def update(idx, include):
            update_vds(idx, include)
            pg_callback(1)

        dataset = query_slice.slice_dataset()
        ds_query = DatasetQuery(dataset, query, progress_callback=update)
        return ds_query.execute()

    try:
        if num_workers == 0:
            index_map = subquery(QuerySlice(0, len(dataset), dataset, query))
        else:
            compute = get_compute_provider(scheduler=scheduler, num_workers=num_workers)

            btch = len(dataset) // num_workers
            subdatasets = [
                QuerySlice(idx * btch, btch, dataset, query)
                for idx in range(0, num_workers)
            ]

            if progressbar:
                result = compute.map_with_progressbar(pg_subquery, subdatasets, total_length=num_samples)  # type: ignore
            else:
                result = compute.map(subquery, subdatasets)  # type: ignore

            index_map = []
            for ls in result:
                index_map.extend(ls)

    except Exception as e:
        dataset._send_query_progress(
            query_text=query,
            query_id=query_id,
            end=True,
            progress=100,
            status="failed",
        )
        raise e
    finally:
        if vds and hasattr(vds_queue, "close"):
            vds_queue.close()
        if compute:
            compute.close()
        _del_counter(query_id)
    dataset._send_query_progress(
        query_text=query,
        query_id=query_id,
        end=True,
        progress=100,
        status="success",
    )
    if vds:
        vds.autoflush = True
        vds_thread.join()
    return index_map
