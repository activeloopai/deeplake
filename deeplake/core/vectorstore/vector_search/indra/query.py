import numpy as np

from typing import List, Optional

from deeplake.core.vectorstore.vector_search.indra import tql_distance_metrics


def create_query_string(
    distance_metric: Optional[str],
    tql_filter: str,
    limit: int,
    order: Optional[str],
    tensor_list: List[str],
):
    """Function for creating a query string from a distance metric, limit and order.

    Args:
        distance_metric (str): distance metric to compute similarity of the query embedding with dataset's embeddings.
        tql_filter (str): Additional filter using TQL syntax.
        limit (int): number of samples to return after the search.
        order (str): Type of data ordering after computing similarity score. Defaults to "ASC".
        tensor_list (List[str]): List of tensors to return data for.


    Returns:
        str: TQL representation of the query string.
    """

    # TODO: BRING THIS BACK AND DELETE IMPLEMENTATION BELOW
    # tql_filter_str = tql_filter if tql_filter == "" else " where " + tql_filter
    # tensor_list_str = ", ".join(tensor_list)
    # order_str = "" if order is None else f" order by score {order}"
    # distance_metric_str = (
    #     "" if distance_metric is None else f", {distance_metric} as score"
    # )

    # return f"select * from (select {tensor_list_str}{distance_metric_str}{tql_filter_str}){order_str} limit {limit}"

    ## TODO: DELETE IMPLEMENTATION BELOW AND BRING BACK IMPLEMENTATION ABOVE

    tql_filter_str = tql_filter if tql_filter == "" else " where " + tql_filter
    tensor_list_str = ", ".join(tensor_list)
    distance_metric_str = (
        "" if distance_metric is None else f", {distance_metric} as score"
    )

    order_str = "" if order is None else f" order by {distance_metric} {order}"
    score_str = "" if order is None else f", score"

    return f"select {tensor_list_str}{score_str} from (select *{distance_metric_str}{tql_filter_str}{order_str} limit {limit})"


def create_query(
    distance_metric: str,
    embedding_tensor: str,
    query_embedding: str,
    tql_filter: str,
    limit: int,
    tensor_list: List[str],
):
    """Function for creating a query string from a distance metric, embeddings, query_embedding, and limit.

    Args:
        distance_metric (str): distance metric to compute similarity of the query embedding with dataset's embeddings.
        embedding_tensor (str): name of the tensor in the dataset with ``htype = "embedding"``.
        query_embedding (str): embedding representation of the query string converted to str.
        tql_filter (str): Additional filter using TQL syntax.
        limit (int): number of samples to return after the search.
        tensor_list (List[str]): List of tensors to return data for.


    Returns:
        str: TQL representation of the query string.
    """

    order = tql_distance_metrics.get_order_type_for_distance_metric(distance_metric)
    tql_distrance_metric = tql_distance_metrics.get_tql_distance_metric(
        distance_metric, embedding_tensor, query_embedding
    )

    query = create_query_string(
        tql_distrance_metric, tql_filter, limit, order, tensor_list
    )
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
    tql_filter: str,
    tensor_list: List[str],
) -> str:
    """Function for converting query_embedding into tql query.

    Args:
        distance_metric (str): distance metric to compute similarity of the query embedding with dataset's embeddings.
        embedding_tensor (str): name of the tensor in the dataset with `htype = "embedding"`.
        query_embedding (np.ndarray]): embedding representation of the query string.
        limit (int): number of samples to return after the search.
        tql_filter (str): Additional filter using TQL syntax.
        tensor_list (list[str]): List of tensors to return data for.


    Returns:
        str: converted tql query string.
    """
    if query_embedding is None:
        return create_query_string(None, tql_filter, limit, None, tensor_list)

    else:
        query_embedding_str = convert_tensor_to_str(query_embedding)

        return create_query(
            distance_metric,
            embedding_tensor,
            query_embedding_str,
            tql_filter,
            limit,
            tensor_list,
        )
