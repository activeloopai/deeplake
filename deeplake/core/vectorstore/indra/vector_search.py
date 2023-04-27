from deeplake.core.vectorstore.indra import query


def vector_search(
    query_embedding, distance_metric, deeplake_dataset, k, embedding_tensor, **kwargs
):
    tql_query = query.parse_query(distance_metric, k, query_embedding, embedding_tensor)
    view = deeplake_dataset.query(tql_query)
    indices = list(view.index.values[0].value)
    scores = view.score.numpy(fetch_chunks=True)
    return indices, scores
