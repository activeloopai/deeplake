from typing import Dict, Union, Callable

from deeplake.core.dataset import Dataset as DeepLakeDataset
from deeplake.core.vectorstore.vector_search import utils
from deeplake.core.vectorstore.vector_search import filter as filter_utils
from deeplake.core import vectorstore


def vector_search(
    query,
    query_emb,
    exec_option,
    dataset,
    logger,
    filter,
    embedding_tensor,
    distance_metric,
    k,
    return_tensors,
    return_view,
    token,
    org_id,
    return_tql,
) -> Union[Dict, DeepLakeDataset]:
    try:
        from indra import api  # type: ignore

        _INDRA_INSTALLED = True  # pragma: no cover
    except ImportError:  # pragma: no cover
        _INDRA_INSTALLED = False  # pragma: no cover

    runtime = utils.get_runtime_from_exec_option(exec_option)

    if callable(filter):
        raise ValueError(
            f"UDF filter functions are not supported with the current `exec_option`={exec_option}. "
        )

    utils.check_indra_installation(exec_option, indra_installed=_INDRA_INSTALLED)

    view, tql_filter = filter_utils.attribute_based_filtering_tql(
        view=dataset,
        filter=filter,
        logger=logger,
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
        return_tensors=return_tensors,
        return_view=return_view,
        token=token,
        org_id=org_id,
        return_tql=return_tql,
    )
