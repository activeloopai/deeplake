from typing import Dict, List, Union
from deeplake.core.dataset import Dataset as DeepLakeDataset

import numpy as np


distance_metric_map = {
    "l2": lambda a, b: np.linalg.norm(a - b, axis=1, ord=2),
    "l1": lambda a, b: np.linalg.norm(a - b, axis=1, ord=1),
    "max": lambda a, b: np.linalg.norm(a - b, axis=1, ord=np.inf),
    "cos": lambda a, b: np.dot(a, b.T)
    / (np.linalg.norm(a) * np.linalg.norm(b, axis=1)),
}


def search(
    deeplake_dataset: DeepLakeDataset,
    query_embedding: np.ndarray,
    embeddings: np.ndarray,
    distance_metric: str = "l2",
    k: int = 4,
) -> Union[Dict, DeepLakeDataset]:
    """Naive search for nearest neighbors
    args:
        deeplake_dataset: DeepLakeDataset,
        query_embedding: np.ndarray
        embeddings: np.ndarray
        k (int): number of nearest neighbors
        return_tensors (List[str]): List of tensors to return. Defaults to None. If None, all tensors are returned.
        distance_metric: distance function 'L2' for Euclidean, 'L1' for Nuclear, 'Max'
            l-infinity distnace, 'cos' for cosine similarity, 'dot' for dot product
    returns:
        Union[Dict, DeepLakeDataset]: Dictionary where keys are tensor names and values are the results of the search
    """

    if embeddings.shape[0] == 0:
        return deeplake_dataset[0:0], []

    else:
        # Calculate the distance between the query_vector and all data_vectors
        distances = distance_metric_map[distance_metric](query_embedding, embeddings)
        nearest_indices = np.argsort(distances)

        nearest_indices = (
            nearest_indices[::-1][:k]
            if distance_metric in ["cos"]
            else nearest_indices[:k]
        )

        return (
            deeplake_dataset[nearest_indices.tolist()],
            distances[nearest_indices].tolist(),
        )
