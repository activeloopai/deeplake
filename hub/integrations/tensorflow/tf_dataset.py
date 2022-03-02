import abc
import functools
import multiprocessing
import sys
import threading
import warnings

import numpy as np
import six
from six.moves import queue as Queue  # pylint: disable=redefined-builtin
from tensorflow.python.util import deprecation
import tensorflow as tf

class HubDataset(tf.data.Dataset):
  """Represents a potentially large set of elements.

  A `Dataset` can be used to represent an input pipeline as a
  collection of elements and a "logical plan" of transformations that act on
  those elements.
  """

  def __init__(self):
    try:
      variant_tensor = self._as_variant_tensor()
    except AttributeError as e:
      if "_as_variant_tensor" in str(e):
        raise AttributeError("Please use `_variant_tensor` instead of "
                             "`_as_variant_tensor()` to obtain the variant "
                             "associated with a dataset.")
      raise AttributeError("{}: A likely cause of this error is that the super "
                           "call for this dataset is not the last line of the "
                           "`__init__` method. The base class invokes the "
                           "`_as_variant_tensor()` method in its constructor "
                           "and if that method uses attributes defined in the "
                           "`__init__` method, those attributes need to be "
                           "defined before the super call.".format(e))
    super(HubDataset, self).__init__(variant_tensor)

  @abc.abstractmethod
  def _as_variant_tensor(self):
    """Creates a scalar `tf.Tensor` of `tf.variant` representing this dataset.

    Returns:
      A scalar `tf.Tensor` of `tf.variant` type, which represents this dataset.
    """
    raise NotImplementedError(f"{type(self)}.as_variant_tensor()")

  @property
  @deprecation.deprecated(
      None, "Use `tf.compat.v1.data.get_output_classes(dataset)`.")
  def output_classes(self):
    """Returns the class of each component of an element of this dataset.

    Returns:
      A (nested) structure of Python `type` objects corresponding to each
      component of an element of this dataset.
    """
    return nest.map_structure(
        lambda component_spec: component_spec._to_legacy_output_classes(),  # pylint: disable=protected-access
        self.element_spec)

  @property
  # @deprecation.deprecated(
  #     None, "Use `tf.compat.v1.data.get_output_shapes(dataset)`.")
  def output_shapes(self):
    """Returns the shape of each component of an element of this dataset.

    Returns:
      A (nested) structure of `tf.TensorShape` objects corresponding to each
      component of an element of this dataset.
    """
    return nest.map_structure(
        lambda component_spec: component_spec._to_legacy_output_shapes(),  # pylint: disable=protected-access
        self.element_spec)

  @property
  @deprecation.deprecated(
      None, "Use `tf.compat.v1.data.get_output_types(dataset)`.")
  def output_types(self):
    """Returns the type of each component of an element of this dataset.

    Returns:
      A (nested) structure of `tf.DType` objects corresponding to each component
      of an element of this dataset.
    """
    return nest.map_structure(
        lambda component_spec: component_spec._to_legacy_output_types(),  # pylint: disable=protected-access
        self.element_spec)

  @property
  def element_spec(self):
    # TODO(b/110122868): Remove this override once all `Dataset` instances
    # implement `element_structure`.
    return structure.convert_legacy_structure(
        self.output_types, self.output_shapes, self.output_classes)

  @staticmethod
  @functools.wraps(tf.data.Dataset.from_tensors)
  def from_tensors(tensors, name=None):
    return DatasetAdapter(tf.data.Dataset.from_tensors(tensors, name=name))

  @staticmethod
  @functools.wraps(tf.data.Dataset.from_tensor_slices)
  def from_tensor_slices(tensors, name=None):
    return DatasetAdapter(tf.data.Dataset.from_tensor_slices(tensors, name=name))

  @staticmethod
  @deprecation.deprecated(None, "Use `tf.data.Dataset.from_tensor_slices()`.")
  def from_sparse_tensor_slices(sparse_tensor):
    """Splits each rank-N `tf.sparse.SparseTensor` in this dataset row-wise.

    Args:
      sparse_tensor: A `tf.sparse.SparseTensor`.

    Returns:
      Dataset: A `Dataset` of rank-(N-1) sparse tensors.
    """
    return DatasetAdapter(tf.data.SparseTensorSliceDataset(sparse_tensor))

  @staticmethod
  @functools.wraps(tf.data.Dataset.from_generator)
  @deprecation.deprecated_args(None, "Use output_signature instead",
                               "output_types", "output_shapes")
  def from_generator(generator,
                     output_types=None,
                     output_shapes=None,
                     args=None,
                     output_signature=None,
                     name=None):
    # Calling Dataset.from_generator with output_shapes or output_types is
    # deprecated, but this is already checked by the decorator on this function.
    with deprecation.silence():
      return DatasetAdapter(
          tf.data.Dataset.from_generator(
              generator,
              output_types,
              output_shapes,
              args,
              output_signature,
              name=name))

  @staticmethod
  @functools.wraps(tf.data.Dataset.range)
  def range(*args, **kwargs):
    return DatasetAdapter(tf.data.Dataset.range(*args, **kwargs))

  @staticmethod
  @functools.wraps(tf.data.Dataset.zip)
  def zip(datasets, name=None):
    return DatasetAdapter(tf.data.Dataset.zip(datasets, name=name))

  @functools.wraps(tf.data.Dataset.concatenate)
  def concatenate(self, dataset, name=None):
    return DatasetAdapter(
        super(HubDataset, self).concatenate(dataset, name=name))

  @functools.wraps(tf.data.Dataset.prefetch)
  def prefetch(self, buffer_size, name=None):
    return DatasetAdapter(
        super(HubDataset, self).prefetch(buffer_size, name=name))

  @staticmethod
  @functools.wraps(tf.data.Dataset.list_files)
  def list_files(file_pattern, shuffle=None, seed=None, name=None):
    return DatasetAdapter(
        tf.data.Dataset.list_files(file_pattern, shuffle, seed, name=name))

  @functools.wraps(tf.data.Dataset.repeat)
  def repeat(self, count=None, name=None):
    return DatasetAdapter(super(HubDataset, self).repeat(count, name=name))

  @functools.wraps(tf.data.Dataset.shuffle)
  def shuffle(self,
              buffer_size,
              seed=None,
              reshuffle_each_iteration=None,
              name=None):
    return DatasetAdapter(
        super(HubDataset, self).shuffle(
            buffer_size, seed, reshuffle_each_iteration, name=name))

  @functools.wraps(tf.data.Dataset.cache)
  def cache(self, filename="", name=None):
    return DatasetAdapter(super(HubDataset, self).cache(filename, name=name))

  @functools.wraps(tf.data.Dataset.take)
  def take(self, count, name=None):
    return DatasetAdapter(super(HubDataset, self).take(count, name=name))

  @functools.wraps(tf.data.Dataset.skip)
  def skip(self, count, name=None):
    return DatasetAdapter(super(HubDataset, self).skip(count, name=name))

  @functools.wraps(tf.data.Dataset.shard)
  def shard(self, num_shards, index, name=None):
    return DatasetAdapter(
        super(HubDataset, self).shard(num_shards, index, name=name))

  @functools.wraps(tf.data.Dataset.batch)
  def batch(self,
            batch_size,
            drop_remainder=False,
            num_parallel_calls=None,
            deterministic=None,
            name=None):
    return DatasetAdapter(
        super(HubDataset, self).batch(
            batch_size,
            drop_remainder,
            num_parallel_calls,
            deterministic,
            name=name))

  @functools.wraps(tf.data.Dataset.padded_batch)
  def padded_batch(self,
                   batch_size,
                   padded_shapes=None,
                   padding_values=None,
                   drop_remainder=False,
                   name=None):
    return DatasetAdapter(
        super(HubDataset, self).padded_batch(
            batch_size,
            padded_shapes,
            padding_values,
            drop_remainder,
            name=name))

  @functools.wraps(tf.data.Dataset.map)
  def map(self,
          map_func,
          num_parallel_calls=None,
          deterministic=None,
          name=None):
    if num_parallel_calls is None:
      return DatasetAdapter(
          tf.data.MapDataset(self, map_func, preserve_cardinality=False))
    else:
      return DatasetAdapter(
          tf.data.ParallelMapDataset(
              self,
              map_func,
              num_parallel_calls,
              deterministic,
              preserve_cardinality=False))

  @functools.wraps(tf.data.Dataset.flat_map)
  def flat_map(self, map_func, name=None):
    return DatasetAdapter(
        super(HubDataset, self).flat_map(map_func, name=name))

  @functools.wraps(tf.data.Dataset.interleave)
  def interleave(self,
                 map_func,
                 cycle_length=None,
                 block_length=None,
                 num_parallel_calls=None,
                 deterministic=None,
                 name=None):
    return DatasetAdapter(
        super(HubDataset, self).interleave(
            map_func,
            cycle_length,
            block_length,
            num_parallel_calls,
            deterministic,
            name=name))

  @functools.wraps(tf.data.Dataset.filter)
  def filter(self, predicate, name=None):
    return DatasetAdapter(super(HubDataset, self).filter(predicate, name=name))

  @functools.wraps(tf.data.Dataset.apply)
  def apply(self, transformation_func):
    return DatasetAdapter(super(HubDataset, self).apply(transformation_func))

  @functools.wraps(tf.data.Dataset.window)
  def window(self, size, shift=None, stride=1, drop_remainder=False, name=None):
    return DatasetAdapter(
        super(HubDataset,
              self).window(size, shift, stride, drop_remainder, name=name))

  @functools.wraps(tf.data.Dataset.unbatch)
  def unbatch(self, name=None):
    return DatasetAdapter(super(HubDataset, self).unbatch(name=name))

  @functools.wraps(tf.data.Dataset.with_options)
  def with_options(self, options, name=None):
    return DatasetAdapter(
        super(HubDataset, self).with_options(options, name=name))

class DatasetAdapter(HubDataset):
  """Wraps a V2 `Dataset` object in the `tf.compat.v1.data.Dataset` API."""

  def __init__(self, dataset):
    self._dataset = dataset
    super(DatasetAdapter, self).__init__()

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

