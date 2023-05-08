from deeplake.core.vectorstore.vector_search.indra import tql_distance_metrics


def create_query_string(distance_metric, limit, order="ASC"):
    return f"select *, {distance_metric} as score ORDER BY {distance_metric} {order} LIMIT {limit}"


def create_query(distance_metric, embeddings, query_embedding, limit):
    order = tql_distance_metrics.get_order_type_for_distance_metric(distance_metric)
    tql_distrance_metric = tql_distance_metrics.get_tql_distance_metric(
        distance_metric, embeddings, query_embedding
    )
    query = create_query_string(tql_distrance_metric, limit, order)
    return query


def convert_tensor_to_str(query_embedding):
    """Function for converting a query embedding to a string

    We need to convert tensor to a string to be able to use tql
    with the query embedding. Here we will assume that query_embedding
    is always 2D and first dimension is always 1. At some point the
    logic should be extended to support queries of different dimensions.
    """
    query_embedding = query_embedding.transpose(1, 0)

    query_embedding_str = ""

    for item in query_embedding:
        query_embedding_str += f"{item[0]}, "

    return f"ARRAY[{query_embedding_str[:-2]}]"


def parse_query(distance_metric, limit, query_embedding, embedding_tensor="embedding"):
    """Function for converting query_embedding into tql query."""
    query_embedding_str = convert_tensor_to_str(query_embedding)
    tql_query = create_query(
        distance_metric, embedding_tensor, query_embedding_str, limit
    )
    return tql_query
