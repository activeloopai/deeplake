import numpy as np

try:
    import tensorflow as tf
    from tensorflow.data import Dataset
except ImportError:
    raise ImportError(
        "TensorFlow is not installed. Please install tensorflow to use this feature."
    )

import deeplake


def _to_tensor_spec(col: deeplake.ColumnDefinition):
    dtype = col.dtype.id
    if dtype == "text":
        dtype = "string"

    shape = col.dtype.shape
    if not shape:
        shape = ()

    return tf.TensorSpec(shape=shape, dtype=dtype)


def _from_dataset(ds: deeplake.Dataset):
    output_signature = []
    column_names = []
    for col in ds.schema.columns:
        column_names.append(col.name)
        output_signature.append(_to_tensor_spec(col))

    def generator():
        for item in ds:
            values = []
            for i, col in enumerate(column_names):
                value = item[col]
                signature = output_signature[i]
                if len(signature.shape.dims) == 0 and hasattr(value, "item"):
                    value = value.item()
                values.append(value)
            yield tuple(values)

    return Dataset.from_generator(generator, output_signature=tuple(output_signature))
