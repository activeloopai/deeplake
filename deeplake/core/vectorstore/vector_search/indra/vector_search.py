from deeplake.core.vectorstore.vector_search.indra import query_parser


def vector_search(
    query_embedding, metric_function, deeplake_dataset, k, embedding_tensor, **kwargs
):
    tql_query = query_parser.query_parser(
        metric_function, k, query_embedding, embedding_tensor
    )
    view = deeplake_dataset.query(tql_query)
    return view
