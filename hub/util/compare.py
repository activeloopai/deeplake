from hub.core.tensor import Union, Tensor
from hub.core.dataset import Dataset
from hub.api.dataset import dataset
from hub.client.log import logger
from hub.util.exceptions import (
    HashesTensorDoesNotExistError,
    TensorDoesNotExistError,
)

import os, glob


def jaccard_similarity(list_1, list_2):
    """Calculated the Jaccard similarity score (also known as Intersection over Union) for the two 
       lists being compared

    Args:
        list_1: First list being compared. 
        list_2: Second list being comared.

    Returns:
        A similarity score that ranges from 0.0 to 1.0. The higher the score, the more similar the lists being compared.
    """
    intersection = len(list(set(list_1).intersection(list_2)))
    union = (len(set(list_1)) + len(set(list_2))) - intersection
    return float(intersection) / union


def compare(
    dataset_1 : Dataset, dataset_2 : Dataset
) -> int:
    """Utility that compares two datasets using hashes stored in "hashes" tensor.
      
    Examples:
        >>> sample = hub.compare(dataset_1, dataset_2) 
        1.0

    Args:
        dataset_1, dataset_2 (Dataset): Datasets being compared

    Returns:
        int: The Jaccard similarity score between the two hashlists being compared. This score ranges from 0.0 to 1.0, with 1.0 being the highest.
    """
    
    if not (dataset_1.hashes or dataset_2.hashes):
        raise HashesTensorDoesNotExistError()
    
    hashlist_1 = dataset_1.hashes.numpy()
    hashlist_2 = dataset_2.hashes.numpy()
    
    # mmh3 produces two 64 bit hashes. We access only one of these for each sample with [:,0] 
    similarity_score = jaccard_similarity(hashlist_1[:,0], hashlist_2[:,0])
    
    logger.info(f"The Jaccard similarity score (on a scale from 0.0 to 1.0) is {similarity_score}")
    
    return similarity_score
