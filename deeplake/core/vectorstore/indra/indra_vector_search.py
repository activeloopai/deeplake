from deeplake.core.vectorstore.indra import query
from deeplake.core.vectorstore import utils


def vector_search(
    query_embedding, distance_metric, deeplake_dataset, k, embedding_tensor, **kwargs
):
    from indra import api

    tql_query = query.parse_query(distance_metric, k, query_embedding, embedding_tensor)
    indra_ds = api.dataset(deeplake_dataset.path)
    # view = indra_ds.query(tql_query, runtime={"db_engine": db_engine})

    view = indra_ds.query(tql_query)
    indices = view.indexes

    scores = view.score.numpy()
    return indices, scores
