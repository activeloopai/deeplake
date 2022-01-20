from typing import Callable, List, Sequence
from uuid import uuid4

import hub

from hub.core.io import SampleStreaming
from hub.core.query.query import DatasetQuery
from hub.util.compute import get_compute_provider
from hub.util.dataset import map_tensor_keys
from hub.util.exceptions import FilterError
from hub.util.hash import hash_inputs


def filter_dataset(
    dataset: hub.Dataset,
    filter_function: Callable[[hub.Dataset], bool],
    num_workers: int = 0,
    scheduler: str = "threaded",
    progressbar: bool = True,
    query_text=None,
) -> hub.Dataset:
    index_map: List[int]

    if num_workers > 0:
        index_map = filter_with_compute(
            dataset,
            filter_function,
            num_workers,
            scheduler,
            progressbar,
            query_text=query_text,
        )
    else:
        index_map = filter_inplace(
            dataset, filter_function, progressbar, query_text=query_text
        )

    return dataset[index_map]  # type: ignore [this is fine]


def filter_with_compute(
    dataset: hub.Dataset,
    filter_function: Callable,
    num_workers: int,
    scheduler: str,
    progressbar: bool = True,
    query_text=None,
) -> List[int]:

    blocks = SampleStreaming(dataset, tensors=map_tensor_keys(dataset)).list_blocks()
    compute = get_compute_provider(scheduler=scheduler, num_workers=num_workers)

    def filter_slice(indices: Sequence[int]):
        result = list()
        for i in indices:
            if filter_function(dataset[i]):
                result.append(i)

        return result

    def pg_filter_slice(pg_callback, indices: Sequence[int]):
        result = list()
        for i in indices:
            if filter_function(dataset[i]):
                result.append(i)
            pg_callback(1)

        return result

    result: Sequence[List[int]]
    idx: List[List[int]] = [block.indices() for block in blocks]

    query_id = (
        str(uuid4().hex)
        if query_text == "UDF"
        else hash_inputs(dataset.path, dataset.pending_commit_id, query_text)
    )
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

    return index_map


def filter_inplace(
    dataset: hub.Dataset, filter_function: Callable, progressbar: bool, query_text=None
) -> List[int]:
    index_map: List[int] = list()

    it = enumerate(dataset)

    if progressbar:
        from tqdm import tqdm  # type: ignore

        it = tqdm(it, total=len(dataset))

    query_id = (
        str(uuid4().hex)
        if query_text == "UDF"
        else hash_inputs(dataset.path, dataset.pending_commit_id, query_text)
    )
    dataset._send_query_progress(
        query_text=query_text, query_id=query_id, start=True, progress=0
    )
    try:
        for i, sample_in in it:
            if filter_function(sample_in):
                index_map.append(i)
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

    return index_map


def query_dataset(
    dataset: hub.Dataset,
    query: str,
    num_workers: int = 0,
    scheduler: str = "threaded",
    progressbar: bool = True,
) -> hub.Dataset:
    index_map: List[int]

    index_map = query_inplace(dataset, query, progressbar, num_workers, scheduler)

    return dataset[index_map]  # type: ignore [this is fine]


def query_inplace(
    dataset: hub.Dataset,
    query: str,
    progressbar: bool,
    num_workers: int,
    scheduler: str,
) -> List[int]:
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

            def update(x: int):
                bar.update(x)

            try:
                ds_query = DatasetQuery(dataset, query, update)
                return ds_query.execute()
            finally:
                bar.close()
        else:
            return DatasetQuery(dataset, query).execute()

    def pg_subquery(pg_callback, query_slice):
        dataset = query_slice.slice_dataset()
        ds_query = DatasetQuery(dataset, query, progress_callback=pg_callback)
        return ds_query.execute()

    if num_workers == 0:
        return subquery(QuerySlice(0, len(dataset), dataset, query))

    compute = get_compute_provider(scheduler=scheduler, num_workers=num_workers)
    try:
        btch = len(dataset) // num_workers
        subdatasets = [
            QuerySlice(idx * btch, btch, dataset, query)
            for idx in range(0, num_workers)
        ]

        if progressbar:
            result = compute.map_with_progressbar(pg_subquery, subdatasets, total_length=len(dataset))  # type: ignore
        else:
            result = compute.map(subquery, subdatasets)  # type: ignore

        index_map = [
            k + dataset_slice.offset
            for x, dataset_slice in zip(result, subdatasets)
            for k in x
        ]  # unfold the result map
    finally:
        compute.close()

    return index_map
