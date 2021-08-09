from hub.core.tensor import Union, Tensor
from hub.core.dataset import Dataset
from hub.api.dataset import dataset
from hub.client.log import logger
from hub.util.exceptions import (
    HashlistDoesNotExistError,
)
from hub.util.keys import (
    hashlist_exists,
)
import os, glob


def load_hashes(src: Union[str, "Tensor"]):
    
    hashlist = []

    # Load all hashes from tensor into list


    return hashlist

def jaccard_similarity(list_1, list_2):
    """Calculated the Jaccard similarity index (also known as Intersection over Union) for the two 
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

def load_hashlist(path: Union[str, Dataset, Tensor]):
    """Returns hashlist from tensor of a dataset 

    Args:
        path (Union[str, Dataset, Tensor]): Path to dataset or Dataset or Tensor for which hashlist will be loaded.

    Returns:
        Hashlist object.
    """
    num_hashlist = 0
    
    if isinstance(path, Tensor):
        hashlist = path.hashlist
    else:
        if isinstance(path, str):
            path_ds = dataset.load(path)
        
        elif isinstance(path, Dataset):
            path_ds = path

        for tensor_name in path_ds.tensors: 
            if hashlist_exists(tensor_name, path.storage):
                num_hashlist += 1
                hashlist = path[tensor_name].hashlist   
    
    if num_hashlist > 1:
        raise NotImplementedError("More than one hashlist found in dataset's tensors. Right now, Hub only supports comparision of datasets containing a single hashlist each." 
        " An alternative is specify the tensors being compared for which hashlist exists, e.g hub.compare(d1.tensor, d2.tensor)")

    if hashlist.is_empty():
        raise HashlistDoesNotExistError

    return hashlist

def compare(
    path_1: Union[Dataset, Tensor], path_2: Union[Dataset, Tensor]
) -> int:
    """Utility that compares hashlist of two different dataset/tensors

    Note: The Jaccard similarity score ranges from 0.0 to 1.0, with 1.0 being the highest.
      
    Examples:
        >>> sample = hub.compare(dataset_1.images, dataset_2.images)
        1.0
        >>> sample = hub.compare(dataset_1, dataset_2) 
        1.0

    Args:
        path_1 (Union[str,Dataset,Tensor]): Dataset/tensor being compared
        path_2 (Union[str,Dataset,Tensor]): Dataset/tensor being compared
    Returns:
        int: The Jaccard similarity index between the two hashlist being compared.
    """

    list_1 = load_hashlist(path_1)
    list_2 = load_hashlist(path_2)

    # Find Jaccard similarity between the two lists
    similarity_score = jaccard_similarity(list_1.hashes, list_2.hashes)

    logger.info(f"The Jaccard similarity score (on a scale from 0.0 to 1.0) is {similarity_score}")
    return similarity_score
