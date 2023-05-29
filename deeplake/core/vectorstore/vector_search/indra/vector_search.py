from typing import Callable

try:
    from indra import api
    
    _INDRA_INSTALLED = True
except ImportError:
    _INDRA_INSTALLED = False

from deeplake.core.vectorstore.vector_search import utils
from deeplake.core.vectorstore.vector_search import filter as filter_utils
from deeplake.core import vectorstore


def vector_search(query, query_emb, exec_option, dataset, logger, filter, embedding_tensor, distance_metric, k):
    runtime = utils.get_runtime_from_exec_option(exec_option)
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
        dataset, logger, filter
    )

    return vectorstore.indra_search_algorithm(
        query_embedding=query_emb,
        distance_metric=distance_metric.lower(),
        deeplake_dataset=view,
        k=k,
        tql_string=query,
        tql_filter=tql_filter,
        embedding_tensor=embedding_tensor,
        runtime=runtime,
    )