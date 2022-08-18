from typing import List, Union, Sequence, Any
from functools import reduce
import numpy as np
from hub.core.linked_sample import LinkedSample
from hub.util.exceptions import TensorDtypeMismatchError
from hub.core.sample import Sample  # type: ignore
import hub


def _get_bigger_dtype(d1, d2):
    if np.can_cast(d1, d2):
        if np.can_cast(d2, d1):
            return d1
        else:
            return d2
    else:
        if np.can_cast(d2, d1):
            return d2
        else:
            return np.object


def get_dtype(val: Union[np.ndarray, Sequence, Sample]) -> np.dtype:
    """Get the dtype of a non-uniform mixed dtype sequence of samples."""

    if hasattr(val, "dtype"):
        return np.dtype(val.dtype)  # type: ignore
    elif isinstance(val, int):
        return np.array(0).dtype
    elif isinstance(val, float):
        return np.array(0.0).dtype
    elif isinstance(val, (str, LinkedSample)):
        return np.array("").dtype
    elif isinstance(val, bool):
        return np.dtype(bool)
    elif isinstance(val, Sequence):
        return reduce(_get_bigger_dtype, map(get_dtype, val))
    else:
        raise TypeError(f"Cannot infer numpy dtype for {val}")


def get_htype(val: Union[np.ndarray, Sequence, Sample]) -> str:
    """Get the htype of a non-uniform mixed dtype sequence of samples."""
    if isinstance(val, hub.core.tensor.Tensor):
        return val.meta.htype
    if hasattr(val, "shape"):  # covers numpy arrays, numpy scalars and hub samples.
        return "generic"
    if isinstance(val, List) and len(val) > 0 and isinstance(val[0], LinkedSample):
        return "generic"
    types = set((map(type, val)))  # type: ignore
    if dict in types:
        return "json"
    if types == set((str,)):
        return "text"
    if object in [  # type: ignore
        np.array(x).dtype if not isinstance(x, np.ndarray) else x.dtype for x in val  # type: ignore
    ]:
        return "json"
    return "generic"


def get_empty_text_like_sample(htype: str):
    """Get an empty sample of the given htype.

    Args:
        htype: htype of the sample.

    Returns:
        Empty sample.

    Raises:
        ValueError: if htype is not one of 'text', 'json', and 'list'.
    """
    if htype == "text":
        return ""
    elif htype == "json":
        return {}
    elif htype == "list":
        return []
    else:
        raise ValueError(
            f"This method should only be used for htypes 'text', 'json' and 'list'. Got {htype}."
        )


def intelligent_cast(
    sample: Any, dtype: Union[np.dtype, str], htype: str
) -> np.ndarray:
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

    return np.array(sample, dtype=dtype)


def get_incompatible_dtype(
    samples: Union[np.ndarray, Sequence], dtype: Union[str, np.dtype]
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
    if isinstance(samples, np.ndarray):
        if samples.size == 0:
            return None
        elif samples.size == 1:
            samples = samples.reshape(1).tolist()[0]

    if isinstance(samples, (int, float, bool)) or hasattr(samples, "dtype"):
        return (
            None
            if np.can_cast(samples, dtype)
            else getattr(samples, "dtype", np.array(samples).dtype)
        )
    elif isinstance(samples, Sequence):
        for dt in map(lambda x: get_incompatible_dtype(x, dtype), samples):
            if dt:
                return dt
        return None
    else:
        raise TypeError(
            f"Unexpected object {samples}. Expected np.ndarray, int, float, bool, str or Sequence."
        )
