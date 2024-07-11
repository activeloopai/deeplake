from deeplake.constants import MB
from deeplake.util.warnings import always_warn

import numpy as np

import uuid
from functools import partial
from typing import Optional, Any, Iterable, List, Dict, Callable, Union


def dp_filter_python(x: dict, filter: Dict) -> bool:
    """Filter helper function for Deep Lake
    For non-dict tensors, perform exact match if target data is not a list, and perform "IN" match if target data is a list.
    For dict tensors, perform exact match for each key-value pair in the target data.
    """

    result = True

    for tensor in filter.keys():
        if result:  # Only evaluate more data if previous filters were True
            data = x[tensor].data()["value"]

            if x[tensor].meta.htype == "json":
                result = result and all(
                    k in data and v == data[k] for k, v in filter[tensor].items()
                )
            else:
                if type(filter[tensor]) == list:
                    result = result and data in filter[tensor]
                else:
                    result = result and data == filter[tensor]

    return result


def attribute_based_filtering_python(
    view, filter: Optional[Union[Dict, Callable]] = None
):
    if len(view) == 0:
        raise ValueError("specified dataset is empty")
    if filter is not None:
        if isinstance(filter, dict):
            for tensor in filter.keys():
                if tensor not in view.tensors:
                    raise ValueError(
                        f"Tensor '{tensor}' is not present in the Vector Store."
                    )  # We keep this check outside of the partial function below in order to not run it on every iteration in the Deep Lake filter

            filter = partial(dp_filter_python, filter=filter)

        view = view.filter(filter)

    return view


def attribute_based_filtering_tql(
    view, filter: Optional[Dict] = None, debug_mode=False, logger=None
):
    """Filter helper function converting filter dictionary to TQL Deep Lake
    For non-dict tensors, perform exact match if target data is not a list, and perform "IN" match if target data is a list.
    For dict tensors, perform exact match for each key-value pair in the target data.
    """

    tql_filter = ""

    if filter is not None:
        if isinstance(filter, dict):
            for tensor in filter.keys():
                if tensor not in view.tensors:
                    raise ValueError(
                        f"Tensor '{tensor}' is not present in the Vector Store."
                    )
                if view[tensor].meta.htype == "json":
                    for key, value in filter[tensor].items():
                        val_str = f"'{value}'" if type(value) == str else f"{value}"
                        tql_filter += f"{tensor}['{key}'] == {val_str} and "
                else:
                    if type(filter[tensor]) == list:
                        val_str = str(filter[tensor])[
                            1:-1
                        ]  # Remove square bracked and add rounded brackets below.

                        tql_filter += f"{tensor} in ({val_str}) and "

                    else:
                        val_str = (
                            f"'{filter[tensor]}'"
                            if isinstance(filter[tensor], str)
                            or isinstance(filter[tensor], np.str_)
                            else f"{filter[tensor]}"
                        )
                        tql_filter += f"{tensor} == {val_str} and "

            tql_filter = tql_filter[:-5]

    if debug_mode and logger is not None:
        logger.warning(f"Converted tql string is: '{tql_filter}'")  # pragma: no cover
    return view, tql_filter


def exact_text_search(view, query):
    view = view.filter(lambda x: query in x["text"].data()["value"])
    scores = [1.0] * len(view)

    if len(view) == 0:
        always_warn(
            "Exact text search wasn't able to find any files. Try other search options like embedding search."
        )
        index = None
    else:
        index = list(view.sample_indices)
    return (view, scores, index)


def get_id_indices(dataset, ids):
    filtered_ids = None
    view = dataset.filter(lambda x: x["ids"].data()["value"] in ids)
    filtered_ids = list(view.sample_indices)

    if len(filtered_ids) != len(ids):
        ids_that_doesnt_exist = get_ids_that_does_not_exist(ids, filtered_ids)
        raise ValueError(
            f"The following ids: {ids_that_doesnt_exist} does not exist in the dataset"
        )
    return filtered_ids


def get_ids_that_does_not_exist(ids, filtered_ids):
    ids_that_doesnt_exist = ""
    for id in ids:
        if id not in filtered_ids:
            ids_that_doesnt_exist += f"`{id}`, "
    return ids_that_doesnt_exist[:-2]


def get_filtered_ids(dataset, filter):
    filtered_ids = None
    view = dataset.filter(partial(dp_filter_python, filter=filter))
    filtered_ids = list(view.sample_indices)
    if len(filtered_ids) == 0:
        raise ValueError(f"{filter} does not exist in the dataset.")
    return filtered_ids


def get_converted_ids(dataset, filter, ids):
    if ids and filter:
        raise ValueError("Either filter or ids should be specified.")

    if ids:
        ids = get_id_indices(dataset, ids)
    else:
        ids = get_filtered_ids(dataset, filter)
    return ids
