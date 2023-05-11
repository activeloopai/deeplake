from typing import Optional, Tuple, List

import numpy as np


distance_metric_map = {
    "l2": lambda a, b: np.linalg.norm(a - b, axis=1, ord=2),
    "l1": lambda a, b: np.linalg.norm(a - b, axis=1, ord=1),
    "max": lambda a, b: np.linalg.norm(a - b, axis=1, ord=np.inf),
    "cos": lambda a, b: np.dot(a, b.T)
    / (np.linalg.norm(a) * np.linalg.norm(b, axis=1)),
}


def vector_search(
    query_embedding: np.ndarray,
    embedding: np.ndarray,
    distance_metric: str = "L2",
    k: Optional[int] = 4,
    **kwargs,
) -> Tuple[List, List]:
    """Naive search for nearest neighbors
    args:
        query_embedding: np.ndarray
        embedding: np.ndarray
        k (int): number of nearest neighbors
        distance_metric: distance function 'L2' for Euclidean, 'L1' for Nuclear, 'Max'
            l-infinity distnace, 'cos' for cosine similarity, 'dot' for dot product
        kwargs: some other optionable parameters.
    returns:
        nearest_indices: List, indices of nearest neighbors
    """
    if embedding.shape[0] == 0:
        return [], []

    # Calculate the distance between the query_vector and all data_vectors
    distances = distance_metric_map[distance_metric](query_embedding, embedding)
    nearest_indices = np.argsort(distances)

    nearest_indices = (
        nearest_indices[::-1][:k] if distance_metric in ["cos"] else nearest_indices[:k]
    )

    return nearest_indices.tolist(), distances[nearest_indices].tolist()
