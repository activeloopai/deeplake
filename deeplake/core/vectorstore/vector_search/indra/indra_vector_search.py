import numpy as np
from typing import Union, List, Any, Optional

from deeplake.core.vectorstore.vector_search.indra import query
from deeplake.core.vectorstore.vector_search import utils
from deeplake.core.dataset import Dataset as DeepLakeDataset


def vector_search(
    query_embedding: Union[List[float], np.ndarray[Any, Any]],
    distance_metric: str,
    deeplake_dataset: DeepLakeDataset,
    k: int,
    embedding_tensor: str,
    **kwargs
):
    """Vector Searching algorithm that uses indra.

    Args:
        query_embedding (Optional[Union[List[float], np.ndarray): embedding representation of the query.
        distance_metric (str): Distance metric to compute similarity between query embedding and dataset embeddings
        deeplake_dataset (DeepLakeDataset): DeepLake dataset object.
        k (int): number of samples to return after the search.
        embedding_tensor (str): name of the tensor in the dataset with `htype = "embedding"`.
        **kwargs (Any): Any additional parameters.
    """
    from indra import api  # type: ignore

    tql_query = query.parse_query(distance_metric, k, query_embedding, embedding_tensor)
    indra_ds = api.dataset(deeplake_dataset.path)

    view = indra_ds.query(tql_query)
    indices = view.indexes

    scores = view.score.numpy()
    return indices, scores
