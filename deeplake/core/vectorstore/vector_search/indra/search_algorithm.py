import numpy as np
from typing import Union, Dict, List, Optional

from deeplake.core.vectorstore.vector_search.indra import query
from deeplake.core.vectorstore.vector_search import utils
from deeplake.core.dataset import Dataset as DeepLakeDataset
from deeplake.enterprise.util import raise_indra_installation_error


def search(
    query_embedding: np.ndarray,
    distance_metric: str,
    deeplake_dataset: DeepLakeDataset,
    k: int,
    tql_string: str,
    tql_filter: str,
    embedding_tensor: str,
    runtime: dict,
    return_tensors: List[str],
    return_view: bool = False,
    deep_memory: bool = False,
    token: Optional[str] = None,
    org_id: Optional[str] = None,
) -> Union[Dict, DeepLakeDataset]:
    """Generalized search algorithm that uses indra. It combines vector search and other TQL queries.

    Args:
        query_embedding (Optional[Union[List[float], np.ndarray): embedding representation of the query.
        distance_metric (str): Distance metric to compute similarity between query embedding and dataset embeddings
        deeplake_dataset (DeepLakeDataset): DeepLake dataset object.
        k (int): number of samples to return after the search.
        tql_string (str): Standalone TQL query for execution without other filters.
        tql_filter (str): Additional filter using TQL syntax
        embedding_tensor (str): name of the tensor in the dataset with `htype = "embedding"`.
        runtime (dict): Runtime parameters for the query.
        return_tensors (List[str]): List of tensors to return data for.
        return_view (bool): Return a Deep Lake dataset view that satisfied the search parameters, instead of a dictinary with data. Defaults to False.
        deep_memory (bool): Use DeepMemory for the search. Defaults to False.
        token (Optional[str], optional): Token used for authentication. Defaults to None.
        org_id (Optional[str], optional): Organization ID, is needed only for local datasets. Defaults to None.

    Raises:
        ValueError: If both tql_string and tql_filter are specified.
        raise_indra_installation_error: If the indra is not installed

    Returns:
        Union[Dict, DeepLakeDataset]: Dictionary where keys are tensor names and values are the results of the search, or a Deep Lake dataset view.
    """
    try:
        from indra import api  # type: ignore

        INDRA_INSTALLED = True
    except ImportError:
        INDRA_INSTALLED = False
        pass

    if tql_string:
        tql_query = tql_string
    else:
        tql_query = query.parse_query(
            distance_metric,
            k,
            query_embedding,
            embedding_tensor,
            tql_filter,
            return_tensors,
        )

    if runtime:
        view, data = deeplake_dataset.query(
            tql_query, runtime=runtime, return_data=True
        )
        return_data = data
    elif deep_memory:
        if not INDRA_INSTALLED:
            raise raise_indra_installation_error(indra_import_error=None)

        from deeplake.enterprise.convert_to_libdeeplake import import_indra_api

        api = import_indra_api()

        indra_dataset = api.dataset(deeplake_dataset.path, token=token, org_id=org_id)
        api.tql.prepare_deepmemory_metrics(indra_dataset)

        indra_view = indra_dataset.query(tql_query)
        indexes = indra_view.indexes
        view = deeplake_dataset[indexes]

        return_data = {}

        for tensor in view.tensors:
            return_data[tensor] = utils.parse_tensor_return(view[tensor])

    else:
        if not INDRA_INSTALLED:
            raise raise_indra_installation_error(
                indra_import_error=None
            )  # pragma: no cover
        return_data = {}

        view = deeplake_dataset.query(
            tql_query,
            runtime=runtime,
        )

        for tensor in view.tensors:
            return_data[tensor] = utils.parse_tensor_return(view[tensor])

    if return_view:
        return view
    return return_data
