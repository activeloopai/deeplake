import numpy as np

from typing import Any, Callable, Dict, List, Union

import deeplake
from deeplake.core.dataset import Dataset as DeepLakeDataset
from deeplake.core import vectorstore


EXEC_OPTION_TO_SEARCH_TYPE: Dict[str, Callable] = {
    "compute_engine": vectorstore.indra_vector_search,
    "python": vectorstore.python_vector_search,
    "tensor_db": vectorstore.remote_engine_vector_search,
}


def search(
    query_embedding: Union[List[float], np.ndarray],
    embedding: Union[List[float], np.ndarray],
    k: int,
    distance_metric: str,
    exec_option: str,
    deeplake_dataset: DeepLakeDataset,
    embedding_tensor: str = "embedding",
):
    """Searching function
    Args:
        query_embedding (Union[List[float], np.ndarray]) - embedding representation of the query
        embedding (Union[List[float], np.ndarray) - full embeddings representation of the dataset, used only in python implementation.
        k (int) - number of samples to return after searching
        distance_metric (str, optional): Type of distance metric to use for sorting the data. Avaliable options are: "L1", "L2", "COS", "MAX".
        exec_option (str, optional): Type of query execution. It could be either "python", "compute_engine" or "tensor_db". Defaults to "python".
            ``python`` - runs on the client and can be used for any data stored anywhere. WARNING: using this option with big datasets is discouraged, because it can lead to some memory issues.
            ``compute_engine`` - runs on the client and can be used for any data stored in or connected to Deep Lake.
            ``tensor_db`` - runs on the Deep Lake Managed Database and can be used for any data stored in the Deep Lake Managed.
        deeplake_dataset (DeepLakeDataset): deeplake dataset object.
        embedding_tensor (str): name of the tensor in the dataset with `htype="embedding"`. Defaults to "embedding".
    """
    return EXEC_OPTION_TO_SEARCH_TYPE[exec_option](
        query_embedding=query_embedding,
        embedding=embedding,
        distance_metric=distance_metric,
        deeplake_dataset=deeplake_dataset,
        k=k,
        embedding_tensor=embedding_tensor,
    )
