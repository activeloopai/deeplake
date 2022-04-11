"""
Helper function for exporting tensorflow dataset wrapper
"""
import warnings

from hub.util.exceptions import (
    ModuleNotInstalledException,
    SampleDecompressionError,
    CorruptedSampleError,
)
from hub.util.check_installation import tensorflow_installed


def dataset_to_tensorflow(dataset, tensors, tobytes):
    """Converts the dataset into a tensorflow compatible format"""
    if not tensorflow_installed():
        raise ModuleNotInstalledException(
            "'tensorflow' should be installed to convert the Dataset into tensorflow format"
        )

    import tensorflow as tf  # type: ignore
    from hub.integrations.tf.hubtensorflowdataset import HubTensorflowDataset  # type: ignore

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
        for index in range(len(dataset)):
            sample = {}
            corrupt_sample_found = False
            for key in tensors:
                try:
                    value = dataset[key][index]
                    if tobytes[key]:
                        value = [value.tobytes()]
                    else:
                        value = value.numpy()
                    sample[key] = value
                except SampleDecompressionError:
                    warnings.warn(
                        f"Skipping corrupt {dataset[key].meta.sample_compression} sample."
                    )
                    corrupt_sample_found = True
            if not corrupt_sample_found:
                yield sample

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
    return HubTensorflowDataset.from_generator(__iter__, output_signature=signature)
