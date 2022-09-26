import numpy as np
from typing import List, Sequence, Union
import warnings

from deeplake.core.chunk.base_chunk import InputSample
from deeplake.core.index.index import Index
from deeplake.util.exceptions import DynamicTensorNumpyError


# used for warning the user if updating a tensor caused suboptimal chunks
CHUNK_UPDATE_WARN_PORTION = 0.2


def format_read_samples(
    samples: List[np.ndarray], index: Index, aslist: bool
) -> Union[np.ndarray, List[np.ndarray]]:
    """Prepares samples being read from the chunk engine in the format the user expects."""

    samples = index.apply(samples)  # type: ignore

    if aslist and all(map(np.isscalar, samples)):
        samples = [arr.item() for arr in samples]

    samples = index.apply_squeeze(samples)  # type: ignore

    if aslist:
        return samples
    else:
        return np.array(samples)


def check_samples_type(samples):
    if not isinstance(samples, (List, np.ndarray)):
        raise TypeError(f"Cannot extend with samples of type {type(samples)}")


def make_sequence(
    samples: Union[np.ndarray, Sequence[InputSample], InputSample], index_length: int
) -> Sequence[InputSample]:
    """Make `samples` a sequence of `InputSample`s.

    Args:
        samples (Union[np.ndarray, Sequence[InputSample]]): Incoming samples to be made into a sequence.
        index_length (int): Number of expected samples in the sequence.

    Raises:
        ValueError: If `index_length` is incompatible with the true length of `samples`.

    Returns:
        Sequence[InputSample]: Sequence of `InputSample`s with the same length as `index_length`.
    """

    if index_length == 1:
        if hasattr(samples, "__len__"):
            if len(samples) != 1:  # type: ignore
                samples = [samples]
        elif hasattr(samples, "shape"):
            if len(samples.shape) > 0 and samples.shape[0] != 1:  # type: ignore
                samples = [samples]
        else:
            samples = [samples]

    if hasattr(samples, "__len__"):
        if index_length != len(samples):  # type: ignore
            raise ValueError(
                f"Index length ({index_length}) and length of samples ({len(samples)}) must be equal for updating a tensor."  # type: ignore
            )
    else:
        samples = [samples]

    return samples  # type: ignore


def check_suboptimal_chunks(
    chunks_nbytes_after_updates: List[int], min_chunk_size: int, max_chunk_size: int
):
    upper_warn_threshold = max_chunk_size * (1 + CHUNK_UPDATE_WARN_PORTION)
    lower_warn_threshold = min_chunk_size * (1 - CHUNK_UPDATE_WARN_PORTION)

    for nbytes in chunks_nbytes_after_updates:
        if nbytes > upper_warn_threshold or nbytes < lower_warn_threshold:
            warnings.warn(
                "After update, some chunks were suboptimal. Be careful when doing lots of updates that modify the sizes of samples by a large amount, these can heavily impact read performance!"
            )
            break


def check_sample_shape(shape, last_shape, key, index, aslist):
    if not aslist and last_shape is not None and shape != last_shape:
        raise DynamicTensorNumpyError(key, index, "shape")
