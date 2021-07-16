from typing import Union, Optional
from hub import Dataset


def dataset_like(
    path: str, like: Union[str, Dataset], like_creds: Optional[dict] = None, **kwargs
) -> Dataset:
    """Create a new dataset from template dataset.

    Args:
        path (str): The full path of the new dataset.
        like (str/Dataset, optional): The full path for the datasets or Dataset object. The new Dataset will have the same tesnors and metas.
        like_creds (dict, optional): A dictionary containing credentials used to access the dataset specified by the 'like' argument.
        **kwargs: Optional keyword arguments for dataset initialization. For more information, check out `hub.Dataset.__init__()`.

    Returns:
        Dataset object with the same structure as `like` dataset.
    """
    new_ds = Dataset(path, **kwargs)
    if isinstance(like, str):
        like = Dataset(like, creds=like_creds)
    for tensor in like.tensors:
        tensor_meta = like[tensor].meta
        new_ds.create_tensor(
            tensor,
            htype=tensor_meta.htype,
            dtype=tensor_meta.dtype,
            sample_compression=tensor_meta.sample_compression,
        )
    return new_ds
