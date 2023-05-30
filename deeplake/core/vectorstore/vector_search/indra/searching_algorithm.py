import numpy as np
from typing import Dict, List

from deeplake.enterprise.convert_to_libdeeplake import dataset_to_libdeeplake

from deeplake.core.vectorstore.vector_search.indra import query
from deeplake.core.vectorstore.vector_search import utils
from deeplake.core.dataset import Dataset as DeepLakeDataset


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
) -> Dict:
    """Vector Searching algorithm that uses indra.

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

    Returns:
        Dict: Dictionary where keys are tensor names and values are the results of the search
    Raises:
        ValueError: If both tql_string and tql_filter are specified.
    """
    from indra import api  # type: ignore

    if tql_string and tql_filter:
        raise ValueError(
            f"tql_string and tql_filter parameters cannot be specified simultaneously."
        )

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
    else:
        return_data = {}

        view = deeplake_dataset.query(
            tql_query,
            runtime=runtime,
        )

        for tensor in view.tensors:
            return_data[tensor] = utils.parse_tensor_return(view[tensor])

    return return_data
