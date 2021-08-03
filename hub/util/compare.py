from hub.core.sample import Sample  # type: ignore
from typing import Optional, Union
from hub.core.dataset import Dataset
from hub.api.dataset import dataset
from hub.core.tensor import Tensor
from hub.util.keys import hashlist_exists
from hub.constants import HASHLIST_FILENAME
from hub.client.log import logger
from hub.util.exceptions import (
    HashlistDoesNotExistError,
)
import os, glob


def jaccard_similarity(list_1, list_2):
    intersection = len(list(set(list_1).intersection(list_2)))
    union = (len(list_1) + len(list_2)) - intersection
    return float(intersection) / union


def compare(
    path_1: Union[str, Dataset, Tensor], path_2: Union[str, Dataset, Tensor]
) -> int:
    """Utility that compares hashlist of two different dataset/tensors

    Examples:
        >>> sample = hub.compare(dataset_1.images, dataset_2.images)
        1.0

    Args:
        path_1 (Union[str,Dataset,Tensor]): Dataset/tensor being compared
        path_2 (Union[str,Dataset,Tensor]): Dataset/tensor being compared
    Returns:
        int: The Jaccard similarity index between the two hashlists of the two tensore being compared.
    """

    if isinstance(path_1, Dataset) or isinstance(path_2, Dataset):
        raise NotImplementedError

    if isinstance(path_1, str) or isinstance(path_2, str):
        raise NotImplementedError

    list_1 = path_1.hashlist
    list_2 = path_2.hashlist

    if list_1.is_empty() or list_2.is_empty():
        raise HashlistDoesNotExistError

    # Find Jaccard similarity between the two lists
    similarity_score = jaccard_similarity(list_1.hashes, list_2.hashes)

    logger.info(f"The Jaccard similarity score is {similarity_score}")
    return similarity_score
