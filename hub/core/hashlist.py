from hub.core.meta.hashlist_meta import HashlistMeta
from hub.core.storage import StorageProvider

from hub.util.keys import get_hashlist_meta_key, hashlist_exists
from hub.util.exceptions import (
    HashlistAlreadyExistsError,
)
from typing import Callable, Dict, Optional, Union, Tuple, List

def create_hashlist(
    key: str,
    storage: StorageProvider,
    **kwargs,
):
    if hashlist_exists(key, storage):
        raise HashlistAlreadyExistsError(key)

    meta_key = get_hashlist_meta_key(key)
    meta = HashlistMeta(
        **kwargs
    )
    storage[meta_key] = meta    


# Find similar datasetes to the current one
def find_similar_datasets():
    pass

# Download hashlists similar to the current one
def get_hashlist():
    pass

# Compare current hashlist to downloaded hashlists
def compare_hashlist(
    key: str,
    storage: StorageProvider,
    **kwargs,
):
    pass

def check_similarity(list1, list2):
    
    c1 = Counter(list1)
    c2 = Counter(list2)

    diff = c1 - c2
    #TODO: Consider c2 - c1 case also 

    common = c1 & c2
    
    c1_length = float(sum(c1.values()))
    c2_length = float(sum(c2.values()))
    diff_length = float(sum(diff.values()))
    common_length = float(sum(common.values()))
    
    if (opt_nomatch):
        print('Non-matching samples: \n', list(diff))

    if (opt_match):
        print('Matching samples: \n', list(common))

    similarity_score = ((common_length / max(c1_length, c2_length))* 100) 
    return similarity_score