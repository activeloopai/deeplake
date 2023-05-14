"""
Helper function for exporting tensorflow dataset wrapper
"""
import warnings

from deeplake.util.exceptions import (
    ModuleNotInstalledException,
    SampleDecompressionError,
    CorruptedSampleError,
    ReadSampleFromChunkError,
)
from deeplake.util.check_installation import tensorflow_installed


def dataset_to_tensorflow(dataset, tensors, tobytes, fetch_chunks=True):
    """Converts the dataset into a tensorflow compatible format"""
    if not tensorflow_installed():
        raise ModuleNotInstalledException(
            "'tensorflow' should be installed to convert the Dataset into tensorflow format"
        )

    import tensorflow as tf  # type: ignore
    from deeplake.integrations.tf.deeplake_tensorflow_dataset import DeepLakeTensorflowDataset  # type: ignore

    if not tensors:
        tensors = dataset.tensors

    if isinstance(tobytes, bool):
        tobytes = {k: tobytes for k in tensors}
    else:
        for k in tobytes:
            if k not in tensors:
                raise Exception(
                    f"Tensor {k} is not present in the list of provided tensors: {tensors}."
                )
        tobytes = {k: k in tobytes for k in tensors}

    def __iter__():
        for sample in dataset:
            out = {}
            corrupt_sample_found = False
            for key in tensors:
                try:
                    value = sample[key]
                    if tobytes[key]:
                        value = [value.tobytes()]
                    else:
                        value = value.numpy(fetch_chunks=fetch_chunks)
                    out[key] = value
                except ReadSampleFromChunkError:
                    warnings.warn(
                        f"Skipping corrupt {dataset[key].meta.sample_compression} sample."
                    )
                    corrupt_sample_found = True
            if not corrupt_sample_found:
                yield out

    def generate_signature():
        signature = {}
        for key in tensors:
            tb = tobytes[key]
            dtype = dataset[key].meta.dtype
            if tb or dtype == "str":
                dtype = tf.string
            shape = (1,) if tb else dataset[key].shape[1:]
            signature[key] = tf.TensorSpec(shape=shape, dtype=dtype)
        return signature

    signature = generate_signature()
    return DeepLakeTensorflowDataset.from_generator(
        __iter__, output_signature=signature
    )
