import deeplake
from deeplake.constants import MB
from deeplake.enterprise.util import raise_indra_installation_error

import numpy as np

import uuid
from functools import partial
from typing import Optional, Any, Iterable, List, Dict, Callable, Union


def dp_filter(x: dict, filter: Dict[str, str]) -> bool:
    """Filter helper function for Deep Lake"""
    metadata = x["metadata"].data()["value"]
    return all(k in metadata and v == metadata[k] for k, v in filter.items())


def attribute_based_filtering(view, filter):
    # attribute based filtering
    if filter is not None:
        if isinstance(filter, dict):
            filter = partial(dp_filter, filter=filter)

        view = view.filter(filter)
        if len(view) == 0:
            return []
    return view


def exact_text_search(view, query):
    view = view.filter(lambda x: query in x["text"].data()["value"])
    scores = [1.0] * len(view)
    index = view.index.values[0].value[0]
    return (view, scores, index)


def get_id_indices(dataset, ids):
    filtered_ids = None
    if ids:
        view = dataset.filter(lambda x: x["ids"].data()["value"] in ids)
        filtered_ids = list(view.sample_indices)

        if len(filtered_ids) != len(ids):
            ids_that_doesnt_exist = get_ids_that_does_not_exist(ids, filtered_ids)
            raise ValueError(
                f"The following ids: {ids_that_doesnt_exist} does not exist in the dataset"
            )
    return filtered_ids or ids


def get_ids_that_does_not_exist(ids, filtered_ids):
    ids_that_doesnt_exist = ""
    for id in ids:
        if id not in filtered_ids:
            ids_that_doesnt_exist += f"`{id}`, "
    return ids_that_doesnt_exist[:-2]


def get_filtered_ids(dataset, filter, ids):
    filtered_ids = None
    if filter:
        # TO DO:
        # 1. check filter with indra
        view = dataset.filter(partial(dp_filter, filter=filter))
        filtered_ids = list(view.sample_indices)
    return filtered_ids or ids
