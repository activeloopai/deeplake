from hub.core.tensor import Union, Tensor
from hub.core.dataset import Dataset
from hub.client.log import logger
from hub.util.exceptions import (
    HashesTensorDoesNotExistError,
)
from hub.constants import HASHES_TENSOR_FOLDER
import os, glob, numpy as np


def jaccard_similarity(list_1, list_2):
    """Calculated the Jaccard similarity score (also known as Intersection over Union) for the two
       lists being compared

    Note:
        Identical samples aren't counted as different when calculating this similarity score. For example, if two datasets contain samples,
        [cat.jpg, cat.jpg, dog.jpg] and [cat.jpg, dog.jpg], the similarity score will still be 1.0.

    Args:
        list_1: First list being compared.
        list_2: Second list being comared.

    Returns:
        A similarity score that ranges from 0.0 to 1.0. The higher the score, the more similar the lists being compared.
    """
    intersection = len(list(set(list_1).intersection(list_2)))
    union = (len(set(list_1)) + len(set(list_2))) - intersection
    return float(intersection) / union


def compare(dataset_1: Dataset, dataset_2: Dataset) -> int:
    """Utility that compares two datasets using hashes stored in "_hashes" tensor.

    Examples:
        >>> sample = hub.compare(dataset_1, dataset_2)
        1.0

    Args:
        dataset_1 (Dataset): Dataset being compared
        dataset_2 (Dataset): Dataset being compared

    Returns:
        int: The Jaccard similarity score between the two hashlists being compared. This score ranges from 0.0 to 1.0, with 1.0 being the highest.

    Raises:
        HashesTensorDoesNotExistError: If hashes tensor doesn't exist in atleast one of the datasets being compared.
    """

    if (
        HASHES_TENSOR_FOLDER not in dataset_1.hidden_tensors
        or HASHES_TENSOR_FOLDER not in dataset_2.hidden_tensors
    ):
        raise HashesTensorDoesNotExistError()

    hashlist_1 = dataset_1[HASHES_TENSOR_FOLDER].numpy()
    hashlist_2 = dataset_2[HASHES_TENSOR_FOLDER].numpy()

    # Concatenating numpy arrays into a single list.
    # For example, the hashlist [[1234], [56], [78]] becomes [1234, 56, 78]
    concat_list_1 = np.concatenate(hashlist_1, axis=None)
    concat_list_2 = np.concatenate(hashlist_2, axis=None)

    similarity_score = jaccard_similarity(concat_list_1, concat_list_2)

    logger.info(
        f"The Jaccard similarity score (on a scale from 0.0 to 1.0) is {similarity_score}"
    )

    return similarity_score
