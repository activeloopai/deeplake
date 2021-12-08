from typing import Callable, List, Sequence

import hub

from hub.core.io import SampleStreaming
from hub.util.compute import get_compute_provider
from hub.util.dataset import map_tensor_keys

import inspect


def filter_dataset(
    dataset: hub.Dataset,
    filter_function: Callable[[hub.Dataset], bool],
    num_workers: int = 0,
    scheduler: str = "threaded",
    progressbar: bool = True,
) -> hub.Dataset:
    index_map: List[int]

    if num_workers > 0:
        index_map = filter_with_compute(
            dataset, filter_function, num_workers, scheduler, progressbar
        )
    else:
        index_map = filter_inplace(dataset, filter_function, progressbar)

    ds = dataset[index_map]
    ds._is_filtered_view = True
    if isinstance(filter_function, hub.core.query.DatasetQuery):
        query = filter_function._query
    else:
        try:
            query = inspect.getsource(filter_function)
        except OSError:
            query = getattr(
                filter_function, "__name__", filter_function.__class__.__name__
            )
    ds._query = query
    return ds  # type: ignore [this is fine]


def filter_with_compute(
    dataset: hub.Dataset,
    filter_function: Callable,
    num_workers: int,
    scheduler: str,
    progressbar: bool = True,
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
            pg_callback(1)
            if filter_function(dataset[i]):
                result.append(i)

        return result

    result: Sequence[List[int]]
    idx: List[List[int]] = [block.indices() for block in blocks]

    try:
        if progressbar:
            result = compute.map_with_progressbar(pg_filter_slice, idx, total_length=len(dataset))  # type: ignore
        else:
            result = compute.map(filter_slice, idx)  # type: ignore
        index_map = [k for x in result for k in x]  # unfold the result map

    finally:
        compute.close()

    return index_map


def filter_inplace(
    dataset: hub.Dataset, filter_function: Callable, progressbar: bool
) -> List[int]:
    index_map: List[int] = list()

    it = enumerate(dataset)

    if progressbar:
        from tqdm import tqdm  # type: ignore

        it = tqdm(it, total=len(dataset))

    for i, sample_in in it:
        if filter_function(sample_in):
            index_map.append(i)

    return index_map
