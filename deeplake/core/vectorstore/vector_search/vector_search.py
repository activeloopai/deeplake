from typing import Dict, Callable, List, Union
from deeplake.core.dataset import Dataset as DeepLakeDataset


import numpy as np

from deeplake.core import vectorstore
from deeplake.core.dataset import Dataset as DeepLakeDataset
from deeplake.core.vectorstore.vector_search import dataset as dataset_utils
from deeplake.core.vectorstore.vector_search import filter as filter_utils


EXEC_OPTION_TO_SEARCH_TYPE: Dict[str, Callable] = {
    "compute_engine": vectorstore.indra_vector_search,
    "python": vectorstore.python_vector_search,
    "tensor_db": vectorstore.indra_vector_search,
}


def search(
    query,
    logger,
    filter,
    query_embedding: Union[List[float], np.ndarray],
    k: int,
    distance_metric: str,
    exec_option: str,
    deeplake_dataset: DeepLakeDataset,
    return_tensors: List[str],
    embedding_tensor: str = "embedding",
    return_view: bool = False,
) -> Union[Dict, DeepLakeDataset]:
    """Searching function
    Args:
        query_embedding (Union[List[float], np.ndarray]) - embedding representation of the query
        k (int) - number of samples to return after searching
        distance_metric (str, optional): Type of distance metric to use for sorting the data. Avaliable options are: "L1", "L2", "COS", "MAX".
        exec_option (str, optional): Type of query execution. It could be either "python", "compute_engine" or "tensor_db". Defaults to "python".
            ``python`` - runs on the client and can be used for any data stored anywhere. WARNING: using this option with big datasets is discouraged, because it can lead to some memory issues.
            ``compute_engine`` - runs on the client and can be used for any data stored in or connected to Deep Lake.
            ``tensor_db`` - runs on the Deep Lake Managed Database and can be used for any data stored in the Deep Lake Managed.
        deeplake_dataset (DeepLakeDataset): deeplake dataset object.
        return_tensors (List[str]): List of tensors to return data for.
        embedding_tensor (str): name of the tensor in the dataset with `htype="embedding"`. Defaults to "embedding".
        return_view (Bool): Return a Deep Lake dataset view that satisfied the search parameters, instead of a dictinary with data. Defaults to False.
    """
    return EXEC_OPTION_TO_SEARCH_TYPE[exec_option](
        query=query,
        query_emb=query_embedding,
        exec_option=exec_option,
        dataset=deeplake_dataset,
        logger=logger,
        filter=filter,
        embedding_tensor=embedding_tensor,
        distance_metric=distance_metric,
        k=k,
        return_tensors=return_tensors,
        return_view=return_view,
    )
