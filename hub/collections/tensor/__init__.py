import dask
import numpy as np

from .core import Tensor


def from_array(array, dtag=None, dcompress=None, chunksize=None) -> Tensor:
    """ Generates tensor from arraylike object
    Parameters
    ----------
    array : np.ndarray
        Numpy array like object with shape, dtype, dims
    dtag : str, optional
        Describes type of the data stored in this array (image, mask, labels, ...)
    dcompress: str, optional
        Argument for compression algorithm, ignore this one, this one does not have any affect yet!
    chunksize:
        Information about how many items (from axis 0) should be stored in the same file if a command is given to save this tensor

    Returns
    -------
    Tensor
        newly generated tensor itself
    """
    meta = {
        "dtype": array.dtype,
        "dtag": dtag,
        "dcompress": dcompress,
        "chunksize": chunksize,
    }
    if str(array.dtype) == "object":
        array = dask.array.from_array(array, chunks=1)
    else:
        array = dask.array.from_array(array)
    return Tensor(meta, array)


def concat(tensors, axis=0, chunksize=-1):
    """ Concats multiple tensors on axis into one tensor
    All input tensors should have same dtag, dtype, dcompress
    """
    raise NotImplementedError()


def stack(tensors, axis=0, chunksize=-1):
    """ Stack multiple tesnors into new axis
    All input tensors should have same dtag, dtype, dcompress
    """
    raise NotImplementedError()


def from_zeros(shape, dtype, dtag=None, dcompress=None, chunksize=-1) -> Tensor:
    """ Generates tensor from 0 filled array
    Parameters
    ----------
    shape : Iterable
        Size of each dimension in list or tuple
    dtype : str
        Data type of array, corresponds to numpy's dtype
    dtag : str, optional
        Describes type of the data stored in this array (image, mask, labels, ...)
    dcompress: str, optional
        Argument for compression algorithm, ignore this one, this one does not have any affect yet!
    chunksize:
        Information about how many items (from axis 0) should be stored in the same file if a command is given to save this tensor

    Returns
    -------
    Tensor
        newly generated tensor itself
    """
    meta = {
        "dtype": dtype,
        "dtag": dtag,
        "dcompress": dcompress,
        "chunksize": chunksize,
    }
    array = dask.array.from_array(np.zeros(shape, dtype))
    return Tensor(meta, array)
