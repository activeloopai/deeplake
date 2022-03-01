import tensorflow as tf
import functools
from tensorflow.python.util import deprecation
from tensorflow.python.data.ops import DatasetV1Adapter

class TensorflowDataset(tf.data.DatasetV2):
    def __init__(self, variant_tensor):
        """Creates a TensorflowDataset object.
        This is a difference between DatasetV1 and DatasetV2. DatasetV1 does not
        take anything in its constructor whereas in the DatasetV2, we expect
        subclasses to create a variant_tensor and pass it in to the super() call.
        Args:
        variant_tensor: A DT_VARIANT tensor that represents the dataset.
        """
        super().__init__(self, variant_tensor)

    def prefetch(self, buffer_size, name=None):
        """Creates a `Dataset` that prefetches elements from this dataset.
        Most dataset input pipelines should end with a call to `prefetch`. This
        allows later elements to be prepared while the current element is being
        processed. This often improves latency and throughput, at the cost of
        using additional memory to store prefetched elements.
        Note: Like other `Dataset` methods, prefetch operates on the
        elements of the input dataset. It has no concept of examples vs. batches.
        `examples.prefetch(2)` will prefetch two elements (2 examples),
        while `examples.batch(20).prefetch(2)` will prefetch 2 elements
        (2 batches, of 20 examples each).
        >>> dataset = tf.data.Dataset.range(3)
        >>> dataset = dataset.prefetch(2)
        >>> list(dataset.as_numpy_iterator())
        [0, 1, 2]
        Args:
        buffer_size: A `tf.int64` scalar `tf.Tensor`, representing the maximum
            number of elements that will be buffered when prefetching. If the value
            `tf.data.AUTOTUNE` is used, then the buffer size is dynamically tuned.
        name: Optional. A name for the tf.data transformation.
        Returns:
        Dataset: A `Dataset`.
        """ 
        print("CHILD PREFETCH")
        return super().prefetch(self, buffer_size, name=name)

    def shuffle(self,
              buffer_size,
              seed=None,
              reshuffle_each_iteration=None,
              name=None):
        """Randomly shuffles the elements of this dataset.
        This dataset fills a buffer with `buffer_size` elements, then randomly
        samples elements from this buffer, replacing the selected elements with new
        elements. For perfect shuffling, a buffer size greater than or equal to the
        full size of the dataset is required.
        For instance, if your dataset contains 10,000 elements but `buffer_size` is
        set to 1,000, then `shuffle` will initially select a random element from
        only the first 1,000 elements in the buffer. Once an element is selected,
        its space in the buffer is replaced by the next (i.e. 1,001-st) element,
        maintaining the 1,000 element buffer.
        `reshuffle_each_iteration` controls whether the shuffle order should be
        different for each epoch. In TF 1.X, the idiomatic way to create epochs
        was through the `repeat` transformation:
        ```python
        dataset = tf.data.Dataset.range(3)
        dataset = dataset.shuffle(3, reshuffle_each_iteration=True)
        dataset = dataset.repeat(2)
        # [1, 0, 2, 1, 2, 0]
        dataset = tf.data.Dataset.range(3)
        dataset = dataset.shuffle(3, reshuffle_each_iteration=False)
        dataset = dataset.repeat(2)
        # [1, 0, 2, 1, 0, 2]
        ```
        In TF 2.0, `tf.data.Dataset` objects are Python iterables which makes it
        possible to also create epochs through Python iteration:
        ```python
        dataset = tf.data.Dataset.range(3)
        dataset = dataset.shuffle(3, reshuffle_each_iteration=True)
        list(dataset.as_numpy_iterator())
        # [1, 0, 2]
        list(dataset.as_numpy_iterator())
        # [1, 2, 0]
        ```
        ```python
        dataset = tf.data.Dataset.range(3)
        dataset = dataset.shuffle(3, reshuffle_each_iteration=False)
        list(dataset.as_numpy_iterator())
        # [1, 0, 2]
        list(dataset.as_numpy_iterator())
        # [1, 0, 2]
        ```
        Args:
        buffer_size: A `tf.int64` scalar `tf.Tensor`, representing the number of
            elements from this dataset from which the new dataset will sample.
        seed: (Optional.) A `tf.int64` scalar `tf.Tensor`, representing the random
            seed that will be used to create the distribution. See
            `tf.random.set_seed` for behavior.
        reshuffle_each_iteration: (Optional.) A boolean, which if true indicates
            that the dataset should be pseudorandomly reshuffled each time it is
            iterated over. (Defaults to `True`.)
        name: (Optional.) A name for the tf.data operation.
        Returns:
        Dataset: A `Dataset`.
        """
        print("CHILD SHUFFLE")
        return super().shuffle(self, buffer_size, seed=None, reshuffle_each_iteration=None, name=None)

    @staticmethod
    @functools.wraps(TensorflowDataset.from_generator)
    @deprecation.deprecated_args(None, "Use output_signature instead",
                               "output_types", "output_shapes")
    def from_generator(generator,
                        output_types=None,
                        output_shapes=None,
                        args=None,
                        output_signature=None,
                        name=None):
        # Calling TensorflowDataset.from_generator with output_shapes or output_types is
        # deprecated, but this is already checked by the decorator on this function.
        with deprecation.silence():
            return DatasetV1Adapter(
                TensorflowDataset.from_generator(
                    generator,
                    output_types,
                    output_shapes,
                    args,
                    output_signature,
                    name=name))

    def from_generator(generator,
                     output_types=None,
                     output_shapes=None,
                     args=None,
                     output_signature=None,
                     name=None):
        """Creates a `Dataset` whose elements are generated by `generator`.

        Note: The current implementation of `Dataset.from_generator()` uses
        `tf.numpy_function` and inherits the same constraints. In particular, it
        requires the dataset and iterator related operations to be placed
        on a device in the same process as the Python program that called
        `Dataset.from_generator()`. In particular, using `from_generator` will
        preclude the use of tf.data service for scaling out dataset processing.
        The body of `generator` will not be serialized in a `GraphDef`, and you
        should not use this method if you need to serialize your model and restore
        it in a different environment.

        The `generator` argument must be a callable object that returns
        an object that supports the `iter()` protocol (e.g. a generator function).

        The elements generated by `generator` must be compatible with either the
        given `output_signature` argument or with the given `output_types` and
        (optionally) `output_shapes` arguments, whichever was specified.

        The recommended way to call `from_generator` is to use the
        `output_signature` argument. In this case the output will be assumed to
        consist of objects with the classes, shapes and types defined by
        `tf.TypeSpec` objects from `output_signature` argument:

        >>> def gen():
        ...   ragged_tensor = tf.ragged.constant([[1, 2], [3]])
        ...   yield 42, ragged_tensor
        >>>
        >>> dataset = tf.data.Dataset.from_generator(
        ...      gen,
        ...      output_signature=(
        ...          tf.TensorSpec(shape=(), dtype=tf.int32),
        ...          tf.RaggedTensorSpec(shape=(2, None), dtype=tf.int32)))
        >>>
        >>> list(dataset.take(1))
        [(<tf.Tensor: shape=(), dtype=int32, numpy=42>,
        <tf.RaggedTensor [[1, 2], [3]]>)]

        There is also a deprecated way to call `from_generator` by either with
        `output_types` argument alone or together with `output_shapes` argument.
        In this case the output of the function will be assumed to consist of
        `tf.Tensor` objects with the types defined by `output_types` and with the
        shapes which are either unknown or defined by `output_shapes`.

        Note: If `generator` depends on mutable global variables or other external
        state, be aware that the runtime may invoke `generator` multiple times
        (in order to support repeating the `Dataset`) and at any time
        between the call to `Dataset.from_generator()` and the production of the
        first element from the generator. Mutating global variables or external
        state can cause undefined behavior, and we recommend that you explicitly
        cache any external state in `generator` before calling
        `Dataset.from_generator()`.

        Note: While the `output_signature` parameter makes it possible to yield
        `Dataset` elements, the scope of `Dataset.from_generator()` should be
        limited to logic that cannot be expressed through tf.data operations. Using
        tf.data operations within the generator function is an anti-pattern and may
        result in incremental memory growth.

        Args:
        generator: A callable object that returns an object that supports the
            `iter()` protocol. If `args` is not specified, `generator` must take no
            arguments; otherwise it must take as many arguments as there are values
            in `args`.
        output_types: (Optional.) A (nested) structure of `tf.DType` objects
            corresponding to each component of an element yielded by `generator`.
        output_shapes: (Optional.) A (nested) structure of `tf.TensorShape`
            objects corresponding to each component of an element yielded by
            `generator`.
        args: (Optional.) A tuple of `tf.Tensor` objects that will be evaluated
            and passed to `generator` as NumPy-array arguments.
        output_signature: (Optional.) A (nested) structure of `tf.TypeSpec`
            objects corresponding to each component of an element yielded by
            `generator`.
        name: (Optional.) A name for the tf.data operations used by
            `from_generator`.

        Returns:
        Dataset: A `Dataset`.
        """
        if not callable(generator):
            raise TypeError("`generator` must be a Python callable.")

        if output_signature is not None:
            if output_types is not None:
                raise TypeError("The `output_types` argument can not be used together "
                            "with the `output_signature` argument.")
            if output_shapes is not None:
                raise TypeError("The `output_shapes` argument can not be used together "
                            "with the `output_signature` argument.")
            for spec in nest.flatten(output_signature):
                if not isinstance(spec, type_spec.TypeSpec):
                    raise TypeError(f"`output_signature` must contain objects that are "
                              f"subclass of `tf.TypeSpec` but found {type(spec)} "
                              f"which is not.")
        else:
            if output_types is None:
                raise TypeError("To specify the output signature you need to provide "
                            "either the `output_signature` argument or the "
                            "`output_types` argument.")

        if output_signature is None:
            if output_shapes is None:
                output_shapes = nest.map_structure(
                lambda _: tensor_shape.TensorShape(None), output_types)
            else:
                output_shapes = nest.map_structure_up_to(output_types,
                                                        tensor_shape.as_shape,
                                                        output_shapes)
            output_signature = nest.map_structure_up_to(output_types,
                                                        tensor_spec.TensorSpec,
                                                        output_shapes, output_types)
        if all(
            isinstance(x, tensor_spec.TensorSpec)
            for x in nest.flatten(output_signature)):
          output_types = nest.pack_sequence_as(
            output_signature, [x.dtype for x in nest.flatten(output_signature)])
          output_shapes = nest.pack_sequence_as(
            output_signature, [x.shape for x in nest.flatten(output_signature)])

        if args is None:
            args = ()
        else:
            args = tuple(ops.convert_n_to_tensor(args, name="args"))

        generator_state = DatasetV2._GeneratorState(generator)

        def get_iterator_id_fn(unused_dummy):
            """Creates a unique `iterator_id` for each pass over the dataset.

            The returned `iterator_id` disambiguates between multiple concurrently
            existing iterators.

            Args:
                unused_dummy: Ignored value.

            Returns:
                A `tf.int64` tensor whose value uniquely identifies an iterator in
                `generator_state`.
            """
            return script_ops.numpy_function(generator_state.get_next_id, args,
                                           dtypes.int64)

        def generator_next_fn(iterator_id_t):
            """Generates the next element from iterator with ID `iterator_id_t`.

            We map this function across an infinite repetition of the
            `iterator_id_t`, and raise `StopIteration` to terminate the iteration.

            Args:
                iterator_id_t: A `tf.int64` tensor whose value uniquely identifies the
                iterator in `generator_state` from which to generate an element.

            Returns:
                The next element to generate from the iterator.
            """
            if output_types and output_shapes:
                flattened_types = [
                    dtypes.as_dtype(dt) for dt in nest.flatten(output_types)
                ]
                flattened_shapes = nest.flatten(output_shapes)

                def generator_py_func(iterator_id):
                    """A `py_func` that will be called to invoke the iterator."""
                    # `next()` raises `StopIteration` when there are no more
                    # elements remaining to be generated.
                    values = next(generator_state.get_iterator(iterator_id))

          # Use the same _convert function from the py_func() implementation to
          # convert the returned values to arrays early, so that we can inspect
          # their values.
          try:
            flattened_values = nest.flatten_up_to(output_types, values)
          except (TypeError, ValueError):
            six.reraise(
                TypeError,
                TypeError(
                    f"`generator` yielded an element that did not match the "
                    f"expected structure. The expected structure was "
                    f"{output_types}, but the yielded element was {values}."),
                sys.exc_info()[2])
          ret_arrays = []
          for ret, dtype in zip(flattened_values, flattened_types):
            try:
              ret_arrays.append(
                  script_ops.FuncRegistry._convert(  # pylint: disable=protected-access
                      ret,
                      dtype=dtype.as_numpy_dtype))
            except (TypeError, ValueError):
              six.reraise(
                  TypeError,
                  TypeError(
                      f"`generator` yielded an element that could not be "
                      f"converted to the expected type. The expected type was "
                      f"{dtype.name}, but the yielded element was {ret}."),
                  sys.exc_info()[2])

          # Additional type and shape checking to ensure that the components of
          # the generated element match the `output_types` and `output_shapes`
          # arguments.
          for (ret_array, expected_dtype,
               expected_shape) in zip(ret_arrays, flattened_types,
                                      flattened_shapes):
            if ret_array.dtype != expected_dtype.as_numpy_dtype:
              raise TypeError(
                  f"`generator` yielded an element of type {ret_array.dtype} "
                  f"where an element of type {expected_dtype.as_numpy_dtype} "
                  f"was expected.")
            if not expected_shape.is_compatible_with(ret_array.shape):
              raise TypeError(
                  f"`generator` yielded an element of shape {ret_array.shape} "
                  f"where an element of shape {expected_shape} was expected.")

          return ret_arrays

        flat_values = script_ops.numpy_function(generator_py_func,
                                                [iterator_id_t],
                                                flattened_types)

        # In debug mode the numpy_function will return a scalar if
        # generator_py_func produces only a single value.
        if not isinstance(flat_values, (list, tuple)):
          flat_values = [flat_values]

        # The `py_func()` op drops the inferred shapes, so we add them back in
        # here.
        if output_shapes is not None:
          for ret_t, shape in zip(flat_values, flattened_shapes):
            ret_t.set_shape(shape)

        return nest.pack_sequence_as(output_types, flat_values)
      else:
        flat_output_types = structure.get_flat_tensor_types(output_signature)

        def generator_py_func(iterator_id):
          """A `py_func` that will be called to invoke the iterator."""
          # `next()` raises `StopIteration` when there are no more
          # elements remaining to be generated.
          values = next(generator_state.get_iterator(iterator_id.numpy()))

          try:
            values = structure.normalize_element(values, output_signature)
          except (TypeError, ValueError):
            six.reraise(
                TypeError,
                TypeError(
                    f"`generator` yielded an element that did not match the "
                    f"expected structure. The expected structure was "
                    f"{output_signature}, but the yielded element was "
                    f"{values}."),
                sys.exc_info()[2])

          values_spec = structure.type_spec_from_value(values)

          if not structure.are_compatible(values_spec, output_signature):
            raise TypeError(
                f"`generator` yielded an element of {values_spec} where an "
                f"element of {output_signature} was expected.")

          return structure.to_tensor_list(output_signature, values)

        return script_ops.eager_py_func(
            generator_py_func, inp=[iterator_id_t], Tout=flat_output_types)

    def finalize_fn(iterator_id_t):
      """Releases host-side state for the iterator with ID `iterator_id_t`."""

      def finalize_py_func(iterator_id):
        generator_state.iterator_completed(iterator_id)
        # We return a dummy value so that the `finalize_fn` has a valid
        # signature.
        # NOTE(mrry): Explicitly create an array of `np.int64` because implicit
        # casting in `py_func()` will create an array of `np.int32` on Windows,
        # leading to a runtime error.
        return np.array(0, dtype=np.int64)

      return script_ops.numpy_function(finalize_py_func, [iterator_id_t],
                                       dtypes.int64)

    # This function associates each traversal of `generator` with a unique
    # iterator ID.
    def flat_map_fn(dummy_arg):
      # The `get_iterator_id_fn` gets a unique ID for the current instance of
      # of the generator.
      # The `generator_next_fn` gets the next element from the iterator with the
      # given ID, and raises StopIteration when that iterator contains no
      # more elements.
      return _GeneratorDataset(
          dummy_arg,
          get_iterator_id_fn,
          generator_next_fn,
          finalize_fn,
          output_signature,
          name=name)

    # A single-element dataset that, each time it is evaluated, contains a
    # freshly-generated and unique (for the returned dataset) int64
    # ID that will be used to identify the appropriate Python state, which
    # is encapsulated in `generator_state`, and captured in
    # `get_iterator_id_map_fn`.
    dummy = 0
    id_dataset = Dataset.from_tensors(dummy, name=name)

    # A dataset that contains all of the elements generated by a
    # single iterator created from `generator`, identified by the
    # iterator ID contained in `id_dataset`. Lifting the iteration
    # into a flat_map here enables multiple repetitions and/or nested
    # versions of the returned dataset to be created, because it forces
    # the generation of a new ID for each version.
    return id_dataset.flat_map(flat_map_fn, name=name)
