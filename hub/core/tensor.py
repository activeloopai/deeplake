from hub.constants import DEFAULT_HTYPE
from hub.core.meta.tensor_meta import TensorMeta

from hub.util.keys import get_tensor_meta_key, tensor_exists
from hub.core.typing import StorageProvider
from hub.util.exceptions import (
    TensorAlreadyExistsError,
)


def create_tensor(
    key: str,
    storage: StorageProvider,
    htype: str = DEFAULT_HTYPE,
    sample_compression: str = None,
    chunk_compression: str = None,
    **kwargs,
):
    """If a tensor does not exist, create a new one with the provided meta.

    Args:
        key (str): Key for where the chunks, index_meta, and tensor_meta will be located in `storage` relative to it's root.
        storage (StorageProvider): StorageProvider that all tensor data is written to.
        htype (str): Htype is how the default tensor metadata is defined.
        sample_compression (str): If a sample is not already compressed (using `hub.load`), the sample will be compressed with `sample_compression`.
            May be `UNCOMPRESSED`, in which case samples are uncompressed before stored.
        chunk_compression (str): Chunk-wise compression has not been implemented yet. # TODO
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
        chunk_compression=chunk_compression,
        **kwargs,
    )
    storage[meta_key] = meta  # type: ignore
