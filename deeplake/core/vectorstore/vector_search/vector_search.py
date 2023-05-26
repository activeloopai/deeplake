from deeplake.core import vectorstore
from deeplake.core.vectorstore.vector_search import dataset as dataset_utils
from deeplake.core.vectorstore.vector_search import filter as filter_utils


def python_vector_search(query, query_emb, exec_option, dataset, logger, embedding_tensor, distance_metric, k):
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

    return vectorstore.python_vector_search(
        deeplake_dataset=view,
        query_embedding=query_emb,
        embeddings=embeddings,
        distance_metric=distance_metric.lower(),
        k=k,
    )


def tql_based_vector_search():
    if type(filter) == Callable:
        raise NotImplementedError(
            f"UDF filter function are not supported with exec_option={exec_option}"
        )
    if query and filter:
        raise NotImplementedError(
            f"query and filter parameters cannot be specified simultaneously."
        )

    utils.check_indra_installation(
        exec_option, indra_installed=_INDRA_INSTALLED
    )

    view, tql_filter = filter_utils.attribute_based_filtering_tql(
        self.dataset, filter
    )

    return vectorstore.vector_search(
        query_embedding=query_emb,
        distance_metric=distance_metric.lower(),
        deeplake_dataset=view,
        k=k,
        tql_string=query,
        tql_filter=tql_filter,
        embedding_tensor=embedding_tensor,
        runtime=runtime,
    )
if exec_option == "python":
    
else:
    