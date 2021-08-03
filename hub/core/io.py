from hub.core.meta.tensor_meta import TensorMeta
from hub.core.sample import Sample
from hub.core.index.index import Index
import numpy as np
from typing import List, Optional, Sequence, Union

SampleValue = Union[np.ndarray, int, float, bool, Sample]


def _get_shape(sample: SampleValue):
    if hasattr(sample, "shape"):
        return sample.shape

    if isinstance(sample, (int, float, bool)):
        return tuple()


def _serialize_input_sample(sample: SampleValue, sample_compression: Optional[str], expected_dtype: np.dtype):
    # TODO: docstring
    # TODO: statictyping

    # TODO: casting?

    if isinstance(sample, Sample):
        raise NotImplementedError

    if sample_compression is not None:
        # TODO: compression?
        raise NotImplementedError

    if isinstance(sample, np.ndarray):
        if sample.dtype != expected_dtype:
            # TODO: use intelligent casting util
            if not np.can_cast(sample.dtype, expected_dtype):
                raise NotImplementedError
            sample = sample.astype(expected_dtype)

        return sample.tobytes()

    if isinstance(sample, (int, float, bool)):
        return bytes(sample)

    raise NotImplementedError


def _check_input_samples_are_valid(buffer_generator, min_chunk_size: int, sample_compression: Optional[str]):
    # TODO: docstring
    # TODO: statictyping

    for buffer, _ in buffer_generator:
        if len(buffer) > min_chunk_size:
            msg = f"Sorry, samples that exceed minimum chunk size ({min_chunk_size} bytes) are not supported yet (coming soon!). Got: {len(buffer)} bytes."
            if sample_compression is None:
                msg += "\nYour data is uncompressed, so setting `sample_compression` in `Dataset.create_tensor` could help here!"
            raise NotImplementedError(msg)


def _make_generator(samples, sample_compression, expected_dtype):
    # TODO: docstring
    # TODO: statictyping

    if isinstance(samples, Sequence):
        for sample in samples:
            buffer = _serialize_input_sample(sample, sample_compression, expected_dtype)
            yield buffer, _get_shape(sample)

    elif isinstance(samples, np.ndarray):
        if sample_compression is None:
            for sample in samples:
                buffer = _serialize_input_sample(sample, sample_compression, expected_dtype)
                yield buffer, _get_shape(sample)

        else:
            raise NotImplementedError
    
    else:
        raise NotImplementedError


def serialize_input_samples(samples: Union[Sequence[SampleValue], SampleValue], meta: TensorMeta, min_chunk_size: int):
    # TODO: docstring
    # TODO: statictyping

    if meta.dtype is None:
        raise ValueError("Dtype must be set before input samples can be serialized.")

    sample_compression = meta.sample_compression
    dtype = np.dtype(meta.dtype)

    # TODO: reuse the same generator or cache the outut values to reduce memory consumption?
    buffer_generator = _make_generator(samples, sample_compression, dtype)
    _check_input_samples_are_valid(buffer_generator, min_chunk_size, dtype)
    return _make_generator(samples, sample_compression, dtype)

    # code from ChunkEngine.extend:
        # if isinstance(samples, np.ndarray):
        #     compression = self.tensor_meta.sample_compression
        #     if compression is None:
        #         buffers = []

        #         # before adding any data, we need to check all sample sizes
        #         for sample in samples:
        #             buffer = memoryview(sample.tobytes())
        #             self._check_sample_size(len(buffer))
        #             buffers.append(buffer)

        #         for buffer in buffers:
        #             self._append_bytes(buffer, sample.shape, sample.dtype)
        #     else:
        #         sample_objects = []
        #         compression = self.tensor_meta.sample_compression

        #         # before adding any data, we need to check all sample sizes
        #         for sample in samples:
        #             sample_object = Sample(array=sample)
        #             sample_objects.append(sample_object)
        #             num_bytes = len(sample_object.compressed_bytes(compression))
        #             self._check_sample_size(num_bytes)

        #         for sample_object in sample_objects:
        #             self.append(sample_object)

        # elif isinstance(samples, Sequence):
        #     if is_uniform_sequence(samples):
        #         self.extend(np.array(samples))
        #     else:
        #         for sample in samples:
        #             self.append(sample)
        # else:
        #     raise TypeError(f"Unsupported type for extending. Got: {type(samples)}")

    # code from chunk engine.append:
        # if isinstance(sample, Sample):
        #     # has to decompress to read the array's shape and dtype
        #     # might be able to optimize this away
        #     compression = self.tensor_meta.sample_compression
        #     data = memoryview(sample.compressed_bytes(compression))
        #     self._check_sample_size(len(data))
        #     self._append_bytes(data, sample.shape, sample.dtype)
        # else:
        #     return self.append(Sample(array=np.array(sample)))

