from hub.core.sample import Sample  # type: ignore
from typing import Optional, Union
from hub.core.dataset import Dataset
from hub.api.dataset import dataset
from hub.core.tensor import Tensor
from hub.util.keys import hashlist_exists
from hub.constants import HASHLIST_FILENAME
import os, glob

def jaccard_similarity(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union

def compare(path1: Union[str, Dataset, Tensor], path2: Union[str, Dataset, Tensor]) -> int:
    """Utility that compares hashlist of two different files

    Note:
        No data is actually loaded until you try to get a property of the returned `Sample`. This is useful for passing along to
            `tensor.append` and `tensor.extend`.

    Examples:
        >>> sample = hub.read("path/to/cat.jpeg")
        >>> type(sample.array)
        <class 'numpy.ndarray'>
        >>> sample.compression
        'jpeg'

    Supported File Types:
        image: png, jpeg, and all others supported by `PIL`: https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#fully-supported-formats

    Args:
        path (str): Path to a supported file.

    Returns:
        Sample: Sample object. Call `sample.array` to get the `np.ndarray`.
    """
    if isinstance(path1, Dataset) or isinstance(path2, Dataset):
        raise NotImplementedError

    if isinstance(path1, str) or isinstance(path2, str):
        raise NotImplementedError

    list1 = path1.hashlist
    list2 = path2.hashlist

    #TODO: Add check to make sure list1 and list2 aren't empty
            
    #Find jaccard similarity between the two lists
    similarity_score = jaccard_similarity(list1.hashes, list2.hashes)
    
    print("Jaccard similarity score: ", similarity_score)
    return similarity_score
