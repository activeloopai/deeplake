from hub.core.sample import Sample  # type: ignore
from typing import Optional, Union
import os

def jaccard_similarity(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union

def find_hashlist(path):
    for root, dirs, files in os.walk(path):
        if 'hashlist.json' in files:
            return os.path.join(root, 'hashlist.json')


# hub.compare(d1.images, d2.images)

# Compare hashlists of two different files     
def compare(path1: Union(str, Dataset), path2: Union(str, Dataset)) -> int:
    """Utility that reads raw data from a file into a `np.ndarray` in 1 line of code. Also provides access to all important metadata.

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
    # Find and load hashlist fro two datasets
    l1 = find_hashlist(path1)
    l2 = find_hashlist(path2)

    #Find jaccard similarity between the two lists
    similarity_score = jaccard_similarity(list1, list2)

    return similarity_score
