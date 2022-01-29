from functools import reduce
from numpy import (
    ndarray,
    dtype as np_dtype,
    array as np_array,
    object as np_object,
    can_cast as np_can_cast
)
from typing import Union, Sequence, Any

import hub
from hub.core.sample import Sample  # type: ignore
from hub.util.exceptions import TensorDtypeMismatchError


def _get_bigger_dtype(d1, d2):
    if np_can_cast(d1, d2):
        if np_can_cast(d2, d1):
            return d1
        else:
            return d2
    else:
        if np_can_cast(d2, d1):
            return d2
        else:
            return np_object


def get_dtype(val: Union[ndarray, Sequence, Sample]) -> np_dtype:
    """Get the dtype of a non-uniform mixed dtype sequence of samples."""

    if hasattr(val, "dtype"):
        return np_dtype(val.dtype)  # type: ignore
    elif isinstance(val, int):
        return np_array(0).dtype
    elif isinstance(val, float):
        return np_array(0.0).dtype
    elif isinstance(val, str):
        return np_array("").dtype
    elif isinstance(val, bool):
        return np_dtype(bool)
    elif isinstance(val, Sequence):
        return reduce(_get_bigger_dtype, map(get_dtype, val))
    else:
        raise TypeError(f"Cannot infer numpy dtype for {val}")


def get_htype(val: Union[ndarray, Sequence, Sample]) -> str:
    """Get the htype of a non-uniform mixed dtype sequence of samples."""
    if isinstance(val, hub.core.tensor.Tensor):
        return val.meta.htype
    if hasattr(val, "shape"):  # covers numpy arrays, numpy scalars and hub samples.
        return "generic"
    types = set((map(type, val)))
    if dict in types:
        return "json"
    if types == set((str,)):
        return "text"
    if np_object in [  # type: ignore
        np_array(x).dtype if not isinstance(x, ndarray) else x.dtype for x in val
    ]:
        return "json"
    return "generic"


def intelligent_cast(
    sample: Any, dtype: Union[np_dtype, str], htype: str
) -> ndarray:
    # TODO: docstring (note: sample can be a scalar)/statictyping
    # TODO: implement better casting here
    if isinstance(sample, Sample):
        sample = sample.array

    if hasattr(sample, "dtype") and sample.dtype == dtype:
        return sample

    err_dtype = get_incompatible_dtype(sample, dtype)
    if err_dtype:
        raise TensorDtypeMismatchError(
            dtype,
            err_dtype,
            htype,
        )

    if hasattr(sample, "astype"):  # covers both ndarrays and scalars
        return sample.astype(dtype)

    return np_array(sample, dtype=dtype)


def get_incompatible_dtype(
    samples: Union[ndarray, Sequence], dtype: Union[str, np_dtype]
):
    """Check if items in a non-uniform mixed dtype sequence of samples can be safely cast to the given dtype.

    Args:
        samples: Sequence of samples
        dtype: dtype to which samples have to be cast

    Returns:
        None if all samples are compatible. If not, the dtype of the offending item is returned.

    Raises:
        TypeError: if samples is of unexepcted type.
    """
    if isinstance(samples, ndarray):
        if samples.size == 0:
            return None
        elif samples.size == 1:
            samples = samples.reshape(1).tolist()[0]

    if isinstance(samples, (int, float, bool)) or hasattr(samples, "dtype"):
        return (
            None
            if np_can_cast(samples, dtype)
            else getattr(samples, "dtype", np_array(samples).dtype)
        )
    elif isinstance(samples, Sequence):
        return all(map(lambda x: get_incompatible_dtype(x, dtype), samples))
    else:
        raise TypeError(
            f"Unexpected object {samples}. Expected np.ndarray, int, float, bool, str or Sequence."
        )
