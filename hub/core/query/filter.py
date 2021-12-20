from typing import Callable, List, Optional, Sequence

import hub

from hub.core.io import SampleStreaming
from hub.util.compute import get_compute_provider
from hub.util.dataset import map_tensor_keys
from time import time

import inspect


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

    if isinstance(filter_function, hub.core.query.DatasetQuery):
        query = filter_function._query
    else:
        try:
            query = inspect.getsource(filter_function)
        except OSError:
            query = getattr(
                filter_function, "__name__", filter_function.__class__.__name__
            )

    vds = dataset._get_empty_vds(result_path, result_ds_args, query=query) if save_result else None

    if num_workers > 0:
        index_map = filter_with_compute(
            dataset,
            filter_function,
            num_workers,
            scheduler,
            progressbar,
            vds,
        )
    else:
        index_map = filter_inplace(
            dataset,
            filter_function,
            progressbar,
            vds,
        )

    ds = dataset[index_map]
    ds._is_filtered_view = True

    ds._query = query
    ds._source_ds_idx = dataset.index.to_json()
    return ds  # type: ignore [this is fine]


def filter_with_compute(
    dataset: hub.Dataset,
    filter_function: Callable,
    num_workers: int,
    scheduler: str,
    progressbar: bool = True,
    vds: Optional[hub.Dataset] = None,
    vds_update_frequency: int = 5,  # seconds
) -> List[int]:

    blocks = SampleStreaming(dataset, tensors=map_tensor_keys(dataset)).list_blocks()
    compute = get_compute_provider(scheduler=scheduler, num_workers=num_workers)

    if vds:
        vds.autoflush = False
        vds.info["total_samples"] = len(dataset)
        vds.info["samples_processed"] = 0

    def filter_slice(indices: Sequence[int]):
        result = list()

        last_update_time = time()
        for i in indices:
            if filter_function(dataset[i]):
                result.append(i)
                if vds:
                    vds.VDS_INDEX.append(i)
                    vds.info["samples_processed"] = vds.info["samples_processed"] + 1
                    if time()  - last_update_time > vds_update_frequency:
                        vds.flush()
                        last_update_time = time()
        if vds:
            vds.autoflush = True
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
    dataset: hub.Dataset,
    filter_function: Callable,
    progressbar: bool,
    vds: Optional[hub.Dataset] = None,
    vds_update_frequency: int = 5,
) -> List[int]:
    index_map: List[int] = list()

    it = enumerate(dataset)

    if vds:
        vds.autoflush = False
        vds.info["total_samples"] = len(dataset)
        vds.info["samples_processed"] = 0


    if progressbar:
        from tqdm import tqdm  # type: ignore

        it = tqdm(it, total=len(dataset))

    for i, sample_in in it:
        if filter_function(sample_in):
            index_map.append(i)
            if vds:
                vds.VDS_INDEX.append(i)
                vds.info["samples_processed"] = vds.info["samples_processed"] + 1
                if time()  - last_update_time > vds_update_frequency:
                    vds.flush()
                    last_update_time = time()

    if vds:
        vds.autoflush = True

    return index_map
