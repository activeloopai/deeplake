"""
Tensorflow dataset wrapper
"""

import abc

import tensorflow as tf  # type: ignore


def deeplake_tf_adapter(_fn):
    """
    Decorator function
    """

    def inner(*args, **kwargs):
        return DatasetAdapter(_fn(*args, **kwargs))

    return inner


# pylint: disable=too-many-public-methods
class DeepLakeTensorflowDataset(tf.data.Dataset):
    """Represents a potentially large set of elements.

    A `DeepLakeTensorflowDataset` can be used to represent an input pipeline as a
    collection of elements and a "logical plan" of transformations that act on
    those elements.
    """

    def __init__(self):
        try:
            variant_tensor = self._as_variant_tensor()
        except AttributeError as attr_ex:
            if "_as_variant_tensor" in str(attr_ex):
                raise AttributeError(
                    "Please use _variant_tensor instead of "
                    "_as_variant_tensor() to obtain the variant "
                    "associated with a dataset"
                ) from attr_ex
        super(__class__, self).__init__(variant_tensor)

    @abc.abstractmethod
    def _as_variant_tensor(self):
        """Creates a scalar `tf.Tensor` of `tf.variant` representing this dataset.
        Returns:
          A scalar `tf.Tensor` of `tf.variant` type, which represents this dataset.
        """
        raise NotImplementedError("Dataset._as_variant_tensor")

    @staticmethod
    @deeplake_tf_adapter
    def from_tensors(tensors):
        return tf.data.Dataset.from_tensors(tensors)

    @staticmethod
    @deeplake_tf_adapter
    def from_tensor_slices(tensors):
        return tf.data.Dataset.from_tensor_slices(tensors)

    @staticmethod
    @deeplake_tf_adapter
    def from_generator(*args, **kwargs):
        return tf.data.Dataset.from_generator(*args, **kwargs)

    @staticmethod
    @deeplake_tf_adapter
    def range(*args, **kwargs):
        return tf.data.Dataset.range(*args, **kwargs)

    @staticmethod
    @deeplake_tf_adapter
    def zip(datasets):
        return tf.data.Dataset.zip(datasets)

    @deeplake_tf_adapter
    def concatenate(self, *args, **kwargs):
        return super(__class__, self).concatenate(*args, **kwargs)

    @deeplake_tf_adapter
    def prefetch(self, *args, **kwargs):
        return super(__class__, self).prefetch(*args, **kwargs)

    @staticmethod
    @deeplake_tf_adapter
    def list_files(*args, **kwargs):
        return tf.data.Dataset.list_files(*args, **kwargs)

    @deeplake_tf_adapter
    def repeat(self, *args, **kwargs):
        return super(__class__, self).repeat(*args, **kwargs)

    @deeplake_tf_adapter
    def shuffle(self, *args, **kwargs):
        return super(__class__, self).shuffle(*args, **kwargs)

    @deeplake_tf_adapter
    def cache(self, *args, **kwargs):
        return super(__class__, self).cache(*args, **kwargs)

    @deeplake_tf_adapter
    def take(self, *args, **kwargs):
        return super(__class__, self).take(*args, **kwargs)

    @deeplake_tf_adapter
    def skip(self, *args, **kwargs):
        return super(__class__, self).skip(*args, **kwargs)

    @deeplake_tf_adapter
    def shard(self, *args, **kwargs):
        return super(__class__, self).shard(*args, **kwargs)

    @deeplake_tf_adapter
    def batch(self, *args, **kwargs):
        return super(__class__, self).batch(*args, **kwargs)

    @deeplake_tf_adapter
    def padded_batch(self, *args, **kwargs):
        return super(__class__, self).padded_batch(*args, **kwargs)

    @deeplake_tf_adapter
    def map(self, *args, **kwargs):
        return super(__class__, self).map(*args, **kwargs)

    @deeplake_tf_adapter
    def flat_map(self, *args, **kwargs):
        return super(__class__, self).flat_map(*args, **kwargs)

    @deeplake_tf_adapter
    def interleave(self, *args, **kwargs):
        return super(__class__, self).interleave(*args, **kwargs)

    @deeplake_tf_adapter
    def filter(self, *args, **kwargs):
        return super(__class__, self).filter(*args, **kwargs)

    @deeplake_tf_adapter
    def apply(self, *args, **kwargs):
        return super(__class__, self).apply(*args, **kwargs)

    @deeplake_tf_adapter
    def window(self, *args, **kwargs):
        return super(__class__, self).window(*args, **kwargs)

    @deeplake_tf_adapter
    def unbatch(self):
        return super(__class__, self).unbatch()

    @deeplake_tf_adapter
    def with_options(self, *args, **kwargs):
        return super(__class__, self).with_options(*args, **kwargs)


class DatasetAdapter(DeepLakeTensorflowDataset):
    """Wraps a V2 `Dataset` object in the `DeepLakeTensorflowDataset` API."""

    # pylint: disable=abstract-method
    def __init__(self, dataset):
        self._dataset = dataset
        super(__class__, self).__init__()

    def _as_variant_tensor(self):
        return self._dataset._variant_tensor  # pylint: disable=protected-access

    def _inputs(self):
        return self._dataset._inputs()  # pylint: disable=protected-access

    def _functions(self):
        return self._dataset._functions()  # pylint: disable=protected-access

    def options(self):
        return self._dataset.options()

    @property
    def element_spec(self):
        return self._dataset.element_spec  # pylint: disable=protected-access

    def __iter__(self):
        return iter(self._dataset)
