from hub.core.meta.tensor_meta import TensorMeta
from hub.core.storage import StorageProvider

from hub.util.keys import get_tensor_meta_key, tensor_exists
from hub.util.exceptions import (
    TensorAlreadyExistsError,
)


def create_tensor(
    key: str,
    storage: StorageProvider,
    htype: str,
    sample_compression: str,
    **kwargs,
):
    """If a tensor does not exist, create a new one with the provided meta.

    Args:
        key (str): Key for where the chunks, index_meta, and tensor_meta will be located in `storage` relative to it's root.
        storage (StorageProvider): StorageProvider that all tensor data is written to.
        htype (str): Htype is how the default tensor metadata is defined.
        sample_compression (str): All samples will be compressed in the provided format. If `None`, samples are uncompressed.
        **kwargs: `htype` defaults can be overridden by passing any of the compatible parameters.
            To see all `htype`s and their correspondent arguments, check out `hub/htypes.py`.

    Raises:
        TensorAlreadyExistsError: If a tensor defined with `key` already exists.
    """

    if tensor_exists(key, storage):
        raise TensorAlreadyExistsError(key)

    meta_key = get_tensor_meta_key(key)
    meta = TensorMeta(
        htype=htype,
        sample_compression=sample_compression,
        **kwargs,
    )
    storage[meta_key] = meta  # type: ignore
