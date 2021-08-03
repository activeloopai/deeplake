from hub.util.exceptions import TensorInvalidSampleShapeError
from hub.util.casting import intelligent_cast
from hub.core.meta.tensor_meta import TensorMeta
from hub.core.sample import Sample, SampleValue  # type: ignore
import numpy as np
from typing import List, Optional, Sequence, Tuple, Union


def _get_shape(sample: SampleValue):
    if hasattr(sample, "shape"):
        return sample.shape

    if isinstance(sample, (int, float, bool)):
        return tuple()

    return (len(sample),)


def _serialize_input_sample(
    sample: SampleValue, sample_compression: Optional[str], expected_dtype: np.dtype
) -> bytes:
    """Converts the incoming sample into a buffer with the proper dtype and compression."""

    if isinstance(sample, Sample):
        if sample.dtype != expected_dtype:
            # TODO: should we cast to the expected dtype? this would require recompression
            # TODO: we should also have a test for this
            raise NotImplementedError

        # only re-compresses when sample_compression doesn't match the original compression
        return sample.compressed_bytes(sample_compression)

    sample = intelligent_cast(np.asarray(sample), expected_dtype)
    if sample_compression is not None:
        return Sample(array=sample).compressed_bytes(sample_compression)
    return sample.tobytes()


def _check_input_samples_are_valid(
    buffer_and_shapes: List, min_chunk_size: int, sample_compression: Optional[str]
):
    """Iterates through all buffers/shapes and raises appropriate errors."""

    expected_dimensionality = None
    for buffer, shape in buffer_and_shapes:
        # check that all samples have the same dimensionality
        if expected_dimensionality is None:
            expected_dimensionality = len(shape)

        if len(buffer) > min_chunk_size:
            msg = f"Sorry, samples that exceed minimum chunk size ({min_chunk_size} bytes) are not supported yet (coming soon!). Got: {len(buffer)} bytes."
            if sample_compression is None:
                msg += "\nYour data is uncompressed, so setting `sample_compression` in `Dataset.create_tensor` could help here!"
            raise NotImplementedError(msg)

        if len(shape) != expected_dimensionality:
            raise TensorInvalidSampleShapeError(shape, expected_dimensionality)


def serialize_input_samples(
    samples: Union[Sequence[SampleValue], SampleValue],
    meta: TensorMeta,
    min_chunk_size: int,
) -> List[Tuple[memoryview, Tuple[int]]]:
    """Casts, compresses, and serializes the incoming samples into a list of buffers and shapes.

    Args:
        samples (Union[Sequence[SampleValue], SampleValue]): Either a single sample or sequence of samples.
        meta (TensorMeta): Tensor meta. Will not be modified.
        min_chunk_size (int): Used to validate that all samples are appropriately sized.

    Raises:
        ValueError: Tensor meta should have it's dtype set.

    Returns:
        List[Tuple[memoryview, Tuple[int]]]: Buffers and their corresponding shapes for the input samples.
    """

    if meta.dtype is None:
        raise ValueError("Dtype must be set before input samples can be serialized.")

    sample_compression = meta.sample_compression
    dtype = np.dtype(meta.dtype)

    serialized = []
    for sample in samples:
        buffer = memoryview(_serialize_input_sample(sample, sample_compression, dtype))
        shape = _get_shape(sample)
        serialized.append((buffer, shape))

    _check_input_samples_are_valid(serialized, min_chunk_size, dtype)
    return serialized
