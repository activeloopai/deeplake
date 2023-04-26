from deeplake.core.vectorstore.indra import query_parser


def vector_search(
    query_embedding, distance_metric, deeplake_dataset, k, embedding_tensor, **kwargs
):
    tql_query = query_parser.query_parser(
        distance_metric, k, query_embedding, embedding_tensor
    )
    view = deeplake_dataset.query(tql_query)
    return view
