from deeplake.core import vectorstore
from deeplake.core.vectorstore.vector_search import dataset as dataset_utils
from deeplake.core.vectorstore.vector_search import filter as filter_utils


def vector_search(query, query_emb, exec_option, dataset, logger, filter, embedding_tensor, distance_metric, k):
    if query is not None:
        raise NotImplementedError(
            f"User-specified TQL queries are not support for exec_option={exec_option} "
        )

    view = filter_utils.attribute_based_filtering_python(dataset, filter)

    embeddings = dataset_utils.fetch_embeddings(
        exec_option=exec_option,
        view=view,
        logger=logger,
        embedding_tensor=embedding_tensor,
    )

    return vectorstore.python_search_algorithm(
        deeplake_dataset=view,
        query_embedding=query_emb,
        embeddings=embeddings,
        distance_metric=distance_metric.lower(),
        k=k,
    )
