from hub.core.meta.hashlist_meta import HashlistMeta
from hub.core.storage import StorageProvider

from hub.util.keys import get_hashlist_meta_key, tensor_exists
from hub.util.exceptions import (
    TensorAlreadyExistsError,
)
from typing import Callable, Dict, Optional, Union, Tuple, List

def create_hashlist(
    key: str,
    storage: StorageProvider,
    **kwargs,
):
    meta_key = get_hashlist_meta_key(key)
    meta = HashlistMeta()

    meta_key = get_hashlist_meta_key(key)
    meta = HashlistMeta(
        **kwargs
    )
    storage[meta_key] = meta
    