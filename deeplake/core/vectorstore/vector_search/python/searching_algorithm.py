from typing import Dict, List, Union
from deeplake.core.dataset import Dataset as DeepLakeDataset
from deeplake.core.vectorstore.vector_search import utils

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
    return_tensors: List[str],
    distance_metric: str = "l2",
    k: int = 4,
    return_view: bool = False,
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
        return_view (Bool): Return a Deep Lake dataset view that satisfied the search parameters, instead of a dictinary with data. Defaults to False.
    returns:
        Union[Dict, DeepLakeDataset]: Dictionary where keys are tensor names and values are the results of the search
    """

    return_data = {}

    if embeddings.shape[0] == 0:
        deeplake_dataset = deeplake_dataset[0:0]

        if return_view:
            return deeplake_dataset
        for tensor in deeplake_dataset.tensors:
            return_data[tensor] = utils.parse_tensor_return(deeplake_dataset[tensor])

        return_data["score"] = []

    else:
        # Calculate the distance between the query_vector and all data_vectors
        distances = distance_metric_map[distance_metric](query_embedding, embeddings)
        nearest_indices = np.argsort(distances)

        nearest_indices = (
            nearest_indices[::-1][:k]
            if distance_metric in ["cos"]
            else nearest_indices[:k]
        )
        if return_view:
            return deeplake_dataset[nearest_indices.tolist()]
        else:
            for tensor in return_tensors:
                return_data[tensor] = utils.parse_tensor_return(
                    deeplake_dataset[tensor][nearest_indices.tolist()]
                )

            return_data["score"] = distances[nearest_indices].tolist()

    return return_data
