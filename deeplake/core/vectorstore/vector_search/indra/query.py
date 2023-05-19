import numpy as np

from typing import Optional, Union, List

from deeplake.core.vectorstore.vector_search.indra import tql_distance_metrics


def create_query_string(distance_metric: str, limit: int, order: str = "ASC"):
    """Function for creating a query string from a distance metric, limit and order.

    Args:
        distance_metric (str): distance metric to compute similarity of the query embedding with dataset's embeddings.
        limit (int): number of samples to return after the search.
        order (str): Type of data ordering after computing similarity score. Defaults to "ASC".

    Returns:
        str: TQL representation of the query string.
    """
    return f"select *, {distance_metric} as score ORDER BY {distance_metric} {order} LIMIT {limit}"


def create_query(
    distance_metric: str,
    embeddings: str,
    query_embedding: str,
    limit: int,
):
    """Function for creating a query string from a distance metric, embeddings, query_embedding, and limit.

    Args:
        distance_metric (str): distance metric to compute similarity of the query embedding with dataset's embeddings.
        embeddings (str): name of the tensor in the dataset with `htype = "embedding"`.
        query_embedding (str): embedding representation of the query string converted to str.
        limit (int): number of samples to return after the search.

    Returns:
        str: TQL representation of the query string.
    """
    order = tql_distance_metrics.get_order_type_for_distance_metric(distance_metric)
    tql_distrance_metric = tql_distance_metrics.get_tql_distance_metric(
        distance_metric, embeddings, query_embedding
    )
    query = create_query_string(tql_distrance_metric, limit, order)
    return query


def convert_tensor_to_str(query_embedding: np.ndarray):
    """Function for converting a query embedding to a string

    We need to convert tensor to a string to be able to use tql
    with the query embedding. Here we will assume that query_embedding
    is always 2D and first dimension is always 1. At some point the
    logic should be extended to support queries of different dimensions.

    Args:
        query_embedding (Union[List[float], np.ndarray]) - embedding representation of the query string.
    """
    if len(query_embedding.shape) > 1:
        query_embedding = query_embedding.transpose(1, 0)
        query_embedding = query_embedding[:, 0]

    query_embedding_str = ""

    for item in query_embedding:
        query_embedding_str += f"{item}, "

    return f"ARRAY[{query_embedding_str[:-2]}]"


def parse_query(
    distance_metric: str,
    limit: int,
    query_embedding: np.ndarray,
    embedding_tensor: str,
) -> str:
    """Function for converting query_embedding into tql query.

    Args:
        distance_metric (str): distance metric to compute similarity of the query embedding with dataset's embeddings.
        embedding_tensor (str): name of the tensor in the dataset with `htype = "embedding"`.
        query_embedding (np.ndarray]): embedding representation of the query string.
        limit (int): number of samples to return after the search.

    Returns:
        str: converted tql query string.
    """
    query_embedding_str = convert_tensor_to_str(query_embedding)
    tql_query = create_query(
        distance_metric, embedding_tensor, query_embedding_str, limit
    )
    return tql_query
