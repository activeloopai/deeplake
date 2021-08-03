from typing import Union, Sequence
import numpy as np



def get_dtype(samples: Union[np.ndarray, Sequence]) -> np.dtype:
    """Get the dtype of a non-uniform mixed dtype sequence of samples."""

    if isinstance(samples, np.ndarray):
        return samples.dtype
    
    if isinstance(samples, (int, float, bool, str)):
        return np.dtype(type(samples))

    if isinstance(samples, Sequence):
        # TODO: instead of just getting the first sample's dtype, maybe we want to check all
        # samples and get the "max"
        return get_dtype(samples[0])

    raise TypeError(f"Unsupported type: {type(samples)}")


def intelligent_cast(sample, dtype) -> np.ndarray:
    # TODO: docstring (note: sample can be a scalar)/statictyping
    # TODO: implement better casting here

    if sample.dtype == dtype:
        return sample

    if not np.can_cast(sample.dtype, dtype):
        raise NotImplementedError(f"Need better casting. From {sample.dtype} -> {dtype}")

    sample = sample.astype(dtype)
    return sample


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
    if isinstance(samples, (int, float, bool, str)) or hasattr(samples, "dtype"):
        return (
            None
            if np.can_cast(samples, dtype)
            else getattr(samples, "dtype", np.array(samples).dtype)
        )
    elif isinstance(samples, Sequence):
        return all(map(lambda x: get_incompatible_dtype(x, dtype), samples))
    else:
        raise TypeError(
            f"Unexpected object {samples}. Expected np.ndarray, int, float, bool, str or Sequence."
        )
