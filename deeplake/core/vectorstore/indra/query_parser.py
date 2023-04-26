from tql_metrics_functions import TQL_METRIC_TO_TQL_QUERY


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
        query_embedding_str += item

    return query_embedding_str


def query_parser(metric_function, limit, query_embedding, embedding_tensor="embedding"):
    """Function for converting query_embedding into tql query."""
    query_embedding_str = convert_tensor_to_str(query_embedding)
    tql_query = TQL_METRIC_TO_TQL_QUERY[metric_function](
        query_embedding_str, embedding_tensor, limit
    )
    return tql_query
