from deeplake.core.vectorstore.indra import query

# from deeplake.enterprise import convert_to_libdeeplake
from indra import api


@profile
def vector_search(
    query_embedding,
    distance_metric,
    deeplake_dataset,
    k,
    embedding_tensor,
    db_engine=False,
    **kwargs
):
    tql_query = query.parse_query(distance_metric, k, query_embedding, embedding_tensor)
    indra_ds = api.dataset(deeplake_dataset.path)
    # view = indra_ds.query(tql_query, runtime={"db_engine": db_engine})

    view = indra_ds.query(tql_query)
    indices = view.indexes

    scores = view.score.numpy()
    return indices, scores
