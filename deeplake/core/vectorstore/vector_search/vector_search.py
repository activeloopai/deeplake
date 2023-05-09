import numpy as np

from typing import Optional, Any, Dict, Callable, Union, List

import deeplake
from deeplake.core.dataset import Dataset as DeepLakeDataset
from deeplake.core import vectorstore

EXEC_OPTION_TO_SEARCH_TYPE = {
    "indra": vectorstore.indra_vector_search,
    "python": vectorstore.python_vector_search,
    "db_engine": vectorstore.remote_engine_vector_search,
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
        exec_option (Optional[str], optional): Type of query execution. It could be either "python", "indra" or "db_engine".
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
