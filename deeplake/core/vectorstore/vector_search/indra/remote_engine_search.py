from deeplake.core.vectorstore.vector_search.indra import query


# TODO: move this logic inside of indra_vector_search.py after supporting vector search queries with deeplake datasets.
def vector_search(
    query_embedding, distance_metric, deeplake_dataset, k, embedding_tensor, **kwargs
):
    tql_query = query.parse_query(distance_metric, k, query_embedding, embedding_tensor)
    indices, scores = deeplake_dataset.query(
        tql_query, runtime={"db_engine": True}, return_indices_and_scores=True
    )
    return indices, scores
