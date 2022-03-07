"""
Tensorflow dataset wrapper
"""
import abc
import functools

import warnings

import six
from tensorflow.python.util import deprecation  # pylint: disable=no-name-in-module
from tensorflow.python.eager import context  # pylint: disable=no-name-in-module

from tensorflow.python.data.ops import dataset_ops  # pylint: disable=no-name-in-module
from tensorflow.python.data.util import traverse  # pylint: disable=no-name-in-module
from tensorflow.python.data.util import nest  # pylint: disable=no-name-in-module
from tensorflow.python.data.util import structure  # pylint: disable=no-name-in-module
from tensorflow.python.data.ops import iterator_ops  # pylint: disable=no-name-in-module

from tensorflow.python.framework import ops  # pylint: disable=no-name-in-module
from tensorflow.python.framework import function  # pylint: disable=no-name-in-module
from tensorflow.python.framework import (
    random_seed as core_random_seed,
)  # pylint: disable=no-name-in-module

import tensorflow as tf


# pylint: disable=too-many-public-methods
class HubTensorflowDataset(tf.data.Dataset):
    """Represents a potentially large set of elements.

    A `Dataset` can be used to represent an input pipeline as a
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
            # pylint: disable=consider-using-f-string
            six.reraise(
                AttributeError,
                "{}: A likely cause of this error is that the super "
                "call for this dataset is not the last line of the "
                "__init__ method. The base class causes the "
                "_as_variant_tensor call in its constructor and "
                "if that uses attributes defined in the __init__ "
                "method, those attrs need to be defined before the "
                "super call.".format(attr_ex),
            )
        super(__class__, self).__init__(variant_tensor)

    @abc.abstractmethod
    def _as_variant_tensor(self):
        """Creates a scalar `tf.Tensor` of `tf.variant` representing this dataset.

        Returns:
          A scalar `tf.Tensor` of `tf.variant` type, which represents this dataset.
        """
        raise NotImplementedError("Dataset._as_variant_tensor")

    @deprecation.deprecated(
        None,
        "This is a deprecated API that should only be used in TF 1 graph "
        "mode and legacy TF 2 graph mode available through `tf.compat.v1`. In "
        "all other situations -- namely, eager mode and inside `tf.function` -- "
        "you can consume dataset elements using `for elem in dataset: ...` or "
        "by explicitly creating iterator via `iterator = iter(dataset)` and "
        "fetching its elements via `values = next(iterator)`. Furthermore, "
        "this API is not available in TF 2. During the transition from TF 1 "
        "to TF 2 you can use `tf.compat.v1.data.make_one_shot_iterator(dataset)` "
        "to create a TF 1 graph mode style iterator for a dataset created "
        "through TF 2 APIs. Note that this should be a transient state of your "
        "code base as there are in general no guarantees about the "
        "interoperability of TF 1 and TF 2 code.",
    )
    def make_one_shot_iterator(self):
        """Creates an iterator for elements of this dataset.

        Note: The returned iterator will be initialized automatically.
        A "one-shot" iterator does not currently support re-initialization. For
        that see `make_initializable_iterator`.

        Example:

        ```python
        # Building graph ...
        dataset = ...
        next_value = dataset.make_one_shot_iterator().get_next()

        # ... from within a session ...
        try:
          while True:
            value = sess.run(next_value)
            ...
        except tf.errors.OutOfRangeError:
            pass
        ```

        Returns:
          An `tf.data.Iterator` for elements of this dataset.
        """
        return self._make_one_shot_iterator()

    # pylint: disable=protected-access
    def _make_one_shot_iterator(self):  # pylint: disable=missing-docstring
        if context.executing_eagerly():
            with ops.colocate_with(self._variant_tensor):
                return iterator_ops.OwnedIterator(self)
        dataset_ops._ensure_same_dataset_graph(self)
        # Some ops (e.g. dataset ops) are marked as stateful but are stil safe to
        # to capture by value. We must allowlist these ops so that the capturing
        # logic captures the ops instead of raising an exception.
        allowlisted_stateful_ops = traverse.obtain_capture_by_value_ops(self)
        graph_level_seed, op_level_seed = core_random_seed.get_seed(None)

        # NOTE(mrry): We capture by value here to ensure that `_make_dataset()` is
        # a 0-argument function.
        @function.Defun(
            capture_by_value=True, allowlisted_stateful_ops=allowlisted_stateful_ops
        )
        def _make_dataset():
            """Factory function for a dataset."""
            # NOTE(mrry): `Defun` does not capture the graph-level seed from the
            # enclosing graph, so if a graph-level seed is present we set the local
            # graph seed based on a combination of the graph- and op-level seeds.
            if graph_level_seed is not None:
                assert op_level_seed is not None
                core_random_seed.set_random_seed(
                    (graph_level_seed + 87654321 * op_level_seed) % (2**63 - 1)
                )

            dataset = self._apply_debug_options()
            return dataset._variant_tensor  # pylint: disable=protected-access

        try:
            _make_dataset.add_to_graph(ops.get_default_graph())
        except ValueError as err:
            if "Cannot capture a stateful node" in str(err):
                # pylint: disable=consider-using-f-string
                raise ValueError(
                    "Failed to create a one-shot iterator for a dataset. "
                    "`Dataset.make_one_shot_iterator()` does not support datasets that "
                    "capture stateful objects, such as a `Variable` or `LookupTable`. "
                    "In these cases, use `Dataset.make_initializable_iterator()`. "
                    "(Original error: %s)" % err
                ) from err
            six.reraise(ValueError, err)

        with ops.colocate_with(self._variant_tensor):
            # pylint: disable=no-member
            return iterator_ops.Iterator(
                ops.gen_dataset_ops.one_shot_iterator(
                    dataset_factory=_make_dataset, **self._flat_structure
                ),
                None,
                tf.data.get_output_types.get_legacy_output_types(self),
                tf.data.get_output_types.get_legacy_output_shapes(self),
                tf.data.get_output_types.get_legacy_output_classes(self),
            )

    @deprecation.deprecated(
        None,
        "This is a deprecated API that should only be used in TF 1 graph "
        "mode and legacy TF 2 graph mode available through `tf.compat.v1`. "
        "In all other situations -- namely, eager mode and inside `tf.function` "
        "-- you can consume dataset elements using `for elem in dataset: ...` "
        "or by explicitly creating iterator via `iterator = iter(dataset)` "
        "and fetching its elements via `values = next(iterator)`. "
        "Furthermore, this API is not available in TF 2. During the transition "
        "from TF 1 to TF 2 you can use "
        "`tf.compat.v1.data.make_initializable_iterator(dataset)` to create a TF "
        "1 graph mode style iterator for a dataset created through TF 2 APIs. "
        "Note that this should be a transient state of your code base as there "
        "are in general no guarantees about the interoperability of TF 1 and TF "
        "2 code.",
    )
    def make_initializable_iterator(self, shared_name=None):
        """Creates an iterator for elements of this dataset.

        Note: The returned iterator will be in an uninitialized state,
        and you must run the `iterator.initializer` operation before using it:

        ```python
        # Building graph ...
        dataset = ...
        iterator = dataset.make_initializable_iterator()
        next_value = iterator.get_next()  # This is a Tensor.

        # ... from within a session ...
        sess.run(iterator.initializer)
        try:
          while True:
            value = sess.run(next_value)
            ...
        except tf.errors.OutOfRangeError:
            pass
        ```

        Args:
          shared_name: (Optional.) If non-empty, the returned iterator will be
            shared under the given name across multiple sessions that share the same
            devices (e.g. when using a remote server).

        Returns:
          A `tf.data.Iterator` for elements of this dataset.

        Raises:
          RuntimeError: If eager execution is enabled.
        """
        return self._make_initializable_iterator(shared_name)

    def _make_initializable_iterator(
        self, shared_name=None
    ):  # pylint: disable=missing-docstring
        if context.executing_eagerly():
            raise RuntimeError(
                "dataset.make_initializable_iterator is not supported when eager "
                "execution is enabled. Use `for element in dataset` instead."
            )
        dataset_ops._ensure_same_dataset_graph(self)
        dataset = self._apply_debug_options()
        if shared_name is None:
            shared_name = ""

        # pylint: disable=no-member
        with ops.colocate_with(self._variant_tensor):
            iterator_resource = ops.gen_dataset_ops.iterator_v2(
                container="", shared_name=shared_name, **self._flat_structure
            )

            initializer = ops.gen_dataset_ops.make_iterator(
                dataset._variant_tensor,  # pylint: disable=protected-access
                iterator_resource,
            )

            return iterator_ops.Iterator(
                iterator_resource,
                initializer,
                tf.data.get_output_types.get_legacy_output_types(dataset),
                tf.data.get_output_types.get_legacy_output_shapes(dataset),
                tf.data.get_output_types.get_legacy_output_classes(dataset),
            )

    @deprecation.deprecated(
        None, "Use `tf.compat.v1.data.get_output_classes(dataset)`."
    )
    # @functools.wraps(tf.data.Dataset.output_classes)
    def output_classes(self):
        """Returns the class of each component of an element of this dataset.

        Returns:
          A (nested) structure of Python `type` objects corresponding to each
          component of an element of this dataset.
        """
        # pylint: disable=protected-access
        return nest.map_structure(
            lambda component_spec: component_spec._to_legacy_output_classes(),
            self.element_spec,
        )

    @deprecation.deprecated(None, "Use `tf.compat.v1.data.get_output_shapes(dataset)`.")
    # @functools.wraps(tf.data.Dataset.output_shapes)
    def output_shapes(self):
        """Returns the shape of each component of an element of this dataset.

        Returns:
          A (nested) structure of `tf.TensorShape` objects corresponding to each
          component of an element of this dataset.
        """
        # pylint: disable=protected-access
        return nest.map_structure(
            lambda component_spec: component_spec._to_legacy_output_shapes(),
            self.element_spec,
        )

    @deprecation.deprecated(None, "Use `tf.compat.v1.data.get_output_types(dataset)`.")
    # @functools.wraps(tf.data.Dataset.output_types)
    def output_types(self):
        """Returns the type of each component of an element of this dataset.

        Returns:
          A (nested) structure of `tf.DType` objects corresponding to each component
          of an element of this dataset.
        """
        # pylint: disable=protected-access
        return nest.map_structure(
            lambda component_spec: component_spec._to_legacy_output_types(),
            self.element_spec,
        )

    @property
    def element_spec(self):
        return structure.convert_legacy_structure(
            self.output_types, self.output_shapes, self.output_classes
        )

    @staticmethod
    @functools.wraps(tf.data.Dataset.from_tensors)
    def from_tensors(tensors):
        return DatasetAdapter(tf.data.Dataset.from_tensors(tensors))

    @staticmethod
    @functools.wraps(tf.data.Dataset.from_tensor_slices)
    def from_tensor_slices(tensors):
        return DatasetAdapter(tf.data.Dataset.from_tensor_slices(tensors))

    @staticmethod
    @deprecation.deprecated(None, "Use `tf.data.Dataset.from_tensor_slices()`.")
    def from_sparse_tensor_slices(sparse_tensor):
        """Splits each rank-N `tf.sparse.SparseTensor` in this dataset row-wise.

        Args:
          sparse_tensor: A `tf.sparse.SparseTensor`.

        Returns:
          Dataset: A `Dataset` of rank-(N-1) sparse tensors.
        """
        return DatasetAdapter(dataset_ops.SparseTensorSliceDataset(sparse_tensor))

    @staticmethod
    @functools.wraps(tf.data.Dataset.from_generator)
    @deprecation.deprecated_args(
        None, "Use output_signature instead", "output_types", "output_shapes"
    )
    def from_generator(
        generator,
        output_types=None,
        output_shapes=None,
        args=None,
        output_signature=None,
    ):
        # Calling tf.data.Dataset.from_generator with output_shapes or output_types is
        # deprecated, but this is already checked by the decorator on this function.
        with deprecation.silence():
            # pylint: disable=not-context-manager
            return DatasetAdapter(
                tf.data.Dataset.from_generator(
                    generator, output_types, output_shapes, args, output_signature
                )
            )

    @staticmethod
    @functools.wraps(tf.data.Dataset.range)
    def range(*args, **kwargs):
        return DatasetAdapter(tf.data.Dataset.range(*args, **kwargs))

    @staticmethod
    @functools.wraps(tf.data.Dataset.zip)
    def zip(datasets):
        return DatasetAdapter(tf.data.Dataset.zip(datasets))

    @functools.wraps(tf.data.Dataset.concatenate)
    def concatenate(self, dataset):
        return DatasetAdapter(super(__class__, self).concatenate(dataset))

    @functools.wraps(tf.data.Dataset.prefetch)
    def prefetch(self, buffer_size):
        return DatasetAdapter(super(__class__, self).prefetch(buffer_size))

    @staticmethod
    @functools.wraps(tf.data.Dataset.list_files)
    def list_files(file_pattern, shuffle=None, seed=None):
        return DatasetAdapter(tf.data.Dataset.list_files(file_pattern, shuffle, seed))

    @functools.wraps(tf.data.Dataset.repeat)
    def repeat(self, count=None):
        return DatasetAdapter(super(__class__, self).repeat(count))

    @functools.wraps(tf.data.Dataset.shuffle)
    def shuffle(self, buffer_size, seed=None, reshuffle_each_iteration=None):
        return DatasetAdapter(
            super(__class__, self).shuffle(buffer_size, seed, reshuffle_each_iteration)
        )

    @functools.wraps(tf.data.Dataset.cache)
    def cache(self, filename=""):
        return DatasetAdapter(super(__class__, self).cache(filename))

    @functools.wraps(tf.data.Dataset.take)
    def take(self, count):
        return DatasetAdapter(super(__class__, self).take(count))

    @functools.wraps(tf.data.Dataset.skip)
    def skip(self, count):
        return DatasetAdapter(super(__class__, self).skip(count))

    @functools.wraps(tf.data.Dataset.shard)
    def shard(self, num_shards, index):
        return DatasetAdapter(super(__class__, self).shard(num_shards, index))

    @functools.wraps(tf.data.Dataset.batch)
    def batch(
        self,
        batch_size,
        drop_remainder=False,
        num_parallel_calls=None,
        deterministic=None,
    ):
        return DatasetAdapter(
            super(__class__, self).batch(
                batch_size, drop_remainder, num_parallel_calls, deterministic
            )
        )

    @functools.wraps(tf.data.Dataset.padded_batch)
    def padded_batch(
        self, batch_size, padded_shapes=None, padding_values=None, drop_remainder=False
    ):
        return DatasetAdapter(
            super(__class__, self).padded_batch(
                batch_size, padded_shapes, padding_values, drop_remainder
            )
        )

    # pylint: disable=no-else-return
    @functools.wraps(tf.data.Dataset.map)
    def map(self, map_func, num_parallel_calls=None, deterministic=None):
        if num_parallel_calls is None:
            return DatasetAdapter(
                dataset_ops.MapDataset(self, map_func, preserve_cardinality=False)
            )
        else:
            return DatasetAdapter(
                dataset_ops.ParallelMapDataset(
                    self,
                    map_func,
                    num_parallel_calls,
                    deterministic,
                    preserve_cardinality=False,
                )
            )

    # pylint: disable=no-else-return
    @deprecation.deprecated(None, "Use `tf.data.Dataset.map()")
    def map_with_legacy_function(
        self, map_func, num_parallel_calls=None, deterministic=None
    ):
        """Maps `map_func` across the elements of this dataset.

        Note: This is an escape hatch for existing uses of `map` that do not work
        with V2 functions. New uses are strongly discouraged and existing uses
        should migrate to `map` as this method will be removed in V2.

        Args:
          map_func: A function mapping a (nested) structure of tensors (having
            shapes and types defined by `self.output_shapes` and
            `self.output_types`) to another (nested) structure of tensors.
          num_parallel_calls: (Optional.) A `tf.int32` scalar `tf.Tensor`,
            representing the number elements to process asynchronously in parallel.
            If not specified, elements will be processed sequentially. If the value
            `tf.data.AUTOTUNE` is used, then the number of parallel
            calls is set dynamically based on available CPU.
          deterministic: (Optional.) When `num_parallel_calls` is specified, this
            boolean controls the order in which the transformation produces
            elements. If set to `False`, the transformation is allowed to yield
            elements out of order to trade determinism for performance. If not
            specified, the `tf.data.Options.experimental_deterministic` option
            (`True` by default) controls the behavior.

        Returns:
          Dataset: A `Dataset`.
        """
        if num_parallel_calls is None:
            if deterministic is not None:
                warnings.warn(
                    "The `deterministic` argument has no effect unless the "
                    "`num_parallel_calls` argument is specified."
                )
            return DatasetAdapter(
                dataset_ops.MapDataset(
                    self, map_func, preserve_cardinality=False, use_legacy_function=True
                )
            )
        else:
            return DatasetAdapter(
                dataset_ops.ParallelMapDataset(
                    self,
                    map_func,
                    num_parallel_calls,
                    deterministic,
                    preserve_cardinality=False,
                    use_legacy_function=True,
                )
            )

    @functools.wraps(tf.data.Dataset.flat_map)
    def flat_map(self, map_func):
        return DatasetAdapter(super(__class__, self).flat_map(map_func))

    # pylint: disable=too-many-arguments
    @functools.wraps(tf.data.Dataset.interleave)
    def interleave(
        self,
        map_func,
        cycle_length=None,
        block_length=None,
        num_parallel_calls=None,
        deterministic=None,
    ):
        return DatasetAdapter(
            super(__class__, self).interleave(
                map_func, cycle_length, block_length, num_parallel_calls, deterministic
            )
        )

    @functools.wraps(tf.data.Dataset.filter)
    def filter(self, predicate):
        return DatasetAdapter(super(__class__, self).filter(predicate))

    @deprecation.deprecated(None, "Use `tf.data.Dataset.filter()")
    def filter_with_legacy_function(self, predicate):
        """Filters this dataset according to `predicate`.

        Note: This is an escape hatch for existing uses of `filter` that do not work
        with V2 functions. New uses are strongly discouraged and existing uses
        should migrate to `filter` as this method will be removed in V2.

        Args:
          predicate: A function mapping a (nested) structure of tensors (having
            shapes and types defined by `self.output_shapes` and
            `self.output_types`) to a scalar `tf.bool` tensor.

        Returns:
          Dataset: The `Dataset` containing the elements of this dataset for which
              `predicate` is `True`.
        """
        return dataset_ops.FilterDataset(self, predicate, use_legacy_function=True)

    @functools.wraps(tf.data.Dataset.apply)
    def apply(self, transformation_func):
        return DatasetAdapter(super(__class__, self).apply(transformation_func))

    @functools.wraps(tf.data.Dataset.window)
    def window(self, size, shift=None, stride=1, drop_remainder=False):
        return DatasetAdapter(
            super(__class__, self).window(size, shift, stride, drop_remainder)
        )

    @functools.wraps(tf.data.Dataset.unbatch)
    def unbatch(self):
        return DatasetAdapter(super(__class__, self).unbatch())

    @functools.wraps(tf.data.Dataset.with_options)
    def with_options(self, options):
        return DatasetAdapter(super(__class__, self).with_options(options))


class DatasetAdapter(HubTensorflowDataset):
    """Wraps a V2 `Dataset` object in the `tf.compat.v1.data.Dataset` API."""

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
