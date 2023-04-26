from typing import Optional, Any, Dict, Callable

from deeplake.core import vectorstore

EXEC_OPTION_TO_SEARCH_TYPE = {
    "indra": vectorstore.indra_vector_search,
    "python": vectorstore.python_vector_search,
}


def search(
    query_embedding: Any,
    embedding: Any,
    k: int,
    distance_metric: str,
    exec_option: Optional[str],
    deeplake_dataset: Any,
    embedding_tensor: str = "embedding",
):
    return EXEC_OPTION_TO_SEARCH_TYPE[exec_option](
        query_embedding=query_embedding,
        embedding=embedding,
        distance_metric=distance_metric,
        deeplake_dataset=deeplake_dataset,
        k=k,
        embedding_tensor=embedding_tensor,
    )
