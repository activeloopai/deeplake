import logging
from typing import Dict, Callable, List, Union, Optional

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
    k: Optional[int],
    exec_option: str,
    deeplake_dataset: DeepLakeDataset,
    distance_metric: str,
    return_tensors: Optional[List[str]] = None,
    query: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
    filter: Optional[Union[Dict, Callable]] = None,
    query_embedding: Optional[Union[List[float], np.ndarray]] = None,
    embedding_tensor: str = "embedding",
    return_view: bool = False,
    deep_memory: bool = False,
    token: Optional[str] = None,
    org_id: Optional[str] = None,
) -> Union[Dict, DeepLakeDataset]:
    """Searching function
    Args:
        query (Optional[str]) - TQL Query string for direct evaluation, without application of additional filters or vector search.
        logger (Optional[logging.Logger]) - logger that will print all of the warnings.
        query_embedding (Union[List[float], np.ndarray]) - embedding representation of the query
        k (int) - number of samples to return after searching
        distance_metric (str, optional): Type of distance metric to use for sorting the data. Avaliable options are: "L1", "L2", "COS", "MAX".
        filter (Union[Dict, Callable], optional): Additional filter evaluated prior to the embedding search.
                - ``Dict`` - Key-value search on tensors of htype json, evaluated on an AND basis (a sample must satisfy all key-value filters to be True) Dict = {"tensor_name_1": {"key": value}, "tensor_name_2": {"key": value}}
                - ``Function`` - Any function that is compatible with `deeplake.filter`.
        exec_option (str, optional): Type of query execution. It could be either "python", "compute_engine" or "tensor_db". Defaults to "python".
            ``python`` - runs on the client and can be used for any data stored anywhere. WARNING: using this option with big datasets is discouraged, because it can lead to some memory issues.
            ``compute_engine`` - runs on the client and can be used for any data stored in or connected to Deep Lake.
            ``tensor_db`` - runs on the Deep Lake Managed Database and can be used for any data stored in the Deep Lake Managed.
        deeplake_dataset (DeepLakeDataset): deeplake dataset object.
        return_tensors (Optional[List[str]], optional): List of tensors to return data for.
        embedding_tensor (str): name of the tensor in the dataset with `htype="embedding"`. Defaults to "embedding".
        return_view (Bool): Return a Deep Lake dataset view that satisfied the search parameters, instead of a dictinary with data. Defaults to False.
        deep_memory (bool): Use DeepMemory for the search. Defaults to False.
        token (Optional[str], optional): Token used for authentication. Defaults to None.
        org_id (Optional[str], optional): Organization ID, is needed only for local datasets. Defaults to None.
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
        deep_memory=deep_memory,
        token=token,
        org_id=org_id,
    )
