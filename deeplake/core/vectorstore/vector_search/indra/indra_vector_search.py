import numpy as np
from typing import Union, List, Any, Optional, Tuple

from deeplake.enterprise.convert_to_libdeeplake import dataset_to_libdeeplake

from deeplake.core.vectorstore.vector_search.indra import query
from deeplake.core.vectorstore.vector_search import utils
from deeplake.core.dataset import Dataset as DeepLakeDataset


def vector_search(
    query_embedding: np.ndarray,
    distance_metric: str,
    deeplake_dataset: DeepLakeDataset,
    k: int,
    embedding_tensor: str,
    **kwargs
) -> Tuple[List, List]:
    """Vector Searching algorithm that uses indra.

    Args:
        query_embedding (Optional[Union[List[float], np.ndarray): embedding representation of the query.
        distance_metric (str): Distance metric to compute similarity between query embedding and dataset embeddings
        deeplake_dataset (DeepLakeDataset): DeepLake dataset object.
        k (int): number of samples to return after the search.
        embedding_tensor (str): name of the tensor in the dataset with `htype = "embedding"`.
        **kwargs (Any): Any additional parameters.

    Returns:
        Tuple[List, List]: tuple representing the indices of the returned embeddings and their respective scores.
    """
    from indra import api  # type: ignore

    tql_query = query.parse_query(distance_metric, k, query_embedding, embedding_tensor)
    indra_ds = dataset_to_libdeeplake(deeplake_dataset)

    view = indra_ds.query(tql_query)
    indices = view.indexes

    scores = view.score.numpy()
    return indices, scores
