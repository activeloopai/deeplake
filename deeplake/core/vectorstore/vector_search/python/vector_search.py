from deeplake.core import vectorstore
from deeplake.core.vectorstore.vector_search import dataset as dataset_utils
from deeplake.core.vectorstore.vector_search import filter as filter_utils
from deeplake.core.vectorstore.vector_search import utils
from deeplake.core.dataset import Dataset as DeepLakeDataset
from typing import Union, Dict


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
    deep_memory,
    token,
    org_id,
) -> Union[Dict, DeepLakeDataset]:
    if query is not None:
        raise NotImplementedError(
            f"User-specified TQL queries are not supported for exec_option={exec_option} "
        )

    view = filter_utils.attribute_based_filtering_python(dataset, filter)

    return_data = {}

    # Only fetch embeddings and run the search algorithm if an embedding query is specified
    if query_emb is not None:
        embeddings = dataset_utils.fetch_embeddings(
            view=view,
            embedding_tensor=embedding_tensor,
        )

        view, scores = vectorstore.python_search_algorithm(
            deeplake_dataset=view,
            query_embedding=query_emb,
            embeddings=embeddings,
            distance_metric=distance_metric.lower(),
            k=k,
        )

        return_data["score"] = scores

    if return_view:
        return view
    else:
        for tensor in return_tensors:
            return_data[tensor] = utils.parse_tensor_return(view[tensor])
        return return_data
