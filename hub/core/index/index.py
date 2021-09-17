from typing import Union, List, Tuple, Iterable, Optional, TypeVar
import numpy as np

IndexValue = Union[int, slice, Tuple[int, ...]]


def slice_has_step(s: slice):
    return (s.step or 1) != 1


def is_trivial_slice(s: slice):
    return not s.start and s.stop is None and not slice_has_step(s)


def is_value_within_slice(s: slice, value: int):
    # TODO: docstring

    if slice_has_step(s):
        # TODO
        raise NotImplementedError

    if is_trivial_slice(s):
        return True

    if s.start is None:
        # needs to be strictly less than
        return value < s.stop

    return (s.start <= value) and (value < s.stop)


def has_negatives(s: slice) -> bool:
    if s.start and s.start < 0:
        return True
    elif s.stop and s.stop < 0:
        return True
    elif s.step and s.step < 0:
        return True
    else:
        return False


def merge_slices(existing_slice: slice, new_slice: slice) -> slice:
    """Compose two slice objects

    Given an iterable x, the following should be equivalent:
    x[existing_slice][new_slice]
    x[merge_slices(existing_slice, new_slice)]

    Args:
        existing_slice (slice): The existing slice to be restricted.
        new_slice (slice): The new slice to be applied to the existing slice.

    Returns:
        slice: the composition of the given slices

    Raises:
        NotImplementedError: Composing slices with negative values is not supported.
            Negative indexing for slices is only supported for the first slice.
    """
    if existing_slice == slice(None):
        return new_slice
    elif new_slice == slice(None):
        return existing_slice

    if has_negatives(existing_slice) or has_negatives(new_slice):
        raise NotImplementedError(
            "Multiple subscripting for slices with negative values is not supported."
        )

    # Combine the steps
    step1 = existing_slice.step if existing_slice.step is not None else 1
    step2 = new_slice.step if new_slice.step is not None else 1
    step = step1 * step2

    # Combine the start points
    start1 = existing_slice.start if existing_slice.start is not None else 0
    start2 = new_slice.start if new_slice.start is not None else 0
    start = start1 + start2 * step1

    # Combine the end points
    stop1 = existing_slice.stop
    stop2 = new_slice.stop

    if stop2 is None:
        stop = stop1
    else:
        stop = start + (stop2 - start2) * step1
        if stop1 is not None:
            stop = min(stop, stop1)

    return slice(start, stop, step)


def slice_at_int(s: slice, i: int):
    """Returns the `i`th element of a slice `s`.

    Examples:
        >>> slice_at_int(slice(None), 10)
        10

        >>> slice_at_int(slice(10, 20, 2), 3)
        16

    Args:
        s (slice): The slice to index into.
        i (int): The integer offset into the slice.

    Returns:
        The index corresponding to the offset into the slice.

    Raises:
        NotImplementedError: Nontrivial slices should not be indexed with negative integers.
    """
    if s == slice(None):
        return i

    if i < 0:
        raise NotImplementedError(
            "Subscripting slices with negative integers is not supported."
        )
    if s.step and s.step < 0:
        return i * s.step - 1
    return (s.start or 0) + i * (s.step or 1)


def slice_length(s: slice, parent_length: int) -> int:
    """Returns the length of a slice given the length of its parent."""
    start, stop, step = s.indices(parent_length)
    step_offset = step - (
        1 if step > 0 else -1
    )  # Used to ceil/floor depending on step direction
    slice_length = stop - start
    total_length = (slice_length + step_offset) // step
    return max(0, total_length)


def tuple_length(t: Tuple[int, ...], l: int) -> int:
    """Returns the length of a tuple of indexes given the length of its parent."""
    return len(t)


class IndexEntry:
    def __init__(self, value: IndexValue = slice(None)):
        self.value = value

    def __getitem__(self, item: IndexValue):
        """Combines the given `item` and this IndexEntry.
        Returns a new IndexEntry representing the composition of the two.

        Examples:
            >>> IndexEntry()[0:100]
            IndexEntry(slice(0, 100, None))

            >>> IndexEntry()[100:200][5]
            IndexEntry(105)

            >>> IndexEntry()[(0, 1, 2, 3)]
            IndexEntry((0, 1, 2, 3))

            >>> IndexEntry()[1, 2, 3]
            IndexEntry((0, 1, 2, 3))

        Args:
            item: The desired sub-index to be composed with this IndexEntry.
                Can be an int, a slice, or a tuple of ints.

        Returns:
            The new IndexEntry object.

        Raises:
            TypeError: An integer IndexEntry should not be indexed further.
        """

        if not self.subscriptable():
            raise TypeError(
                "Subscripting IndexEntry after 'int' is not allowed. Use Index instead."
            )
        elif isinstance(self.value, slice):
            if isinstance(item, int):
                new_value = slice_at_int(self.value, item)
                return IndexEntry(new_value)
            elif isinstance(item, slice):
                return IndexEntry(merge_slices(self.value, item))
            elif isinstance(item, tuple):
                new_value = tuple(slice_at_int(self.value, idx) for idx in item)
                return IndexEntry(new_value)
        elif isinstance(self.value, tuple):
            if isinstance(item, int) or isinstance(item, slice):
                return IndexEntry(self.value[item])
            elif isinstance(item, tuple):
                new_value = tuple(self.value[idx] for idx in item)
                return IndexEntry(new_value)

        raise TypeError(f"Value {item} is of unrecognized type {type(item)}.")

    def subscriptable(self):
        """Returns whether an IndexEntry can be further subscripted."""
        return not isinstance(self.value, int)

    def indices(self, length: int):
        """Generates the sequence of integer indices for a target of a given length."""
        parse_int = lambda i: i if i >= 0 else length + i

        if isinstance(self.value, int):
            yield parse_int(self.value)
        elif isinstance(self.value, slice):
            yield from range(*self.value.indices(length))
        elif isinstance(self.value, tuple):
            yield from map(parse_int, self.value)

    def is_trivial(self):
        """Checks if an IndexEntry represents the entire slice"""

        if not isinstance(self.value, slice):
            return False

        return is_trivial_slice(self.value)

    def length(self, parent_length: int) -> int:
        """Returns the length of an IndexEntry given the length of the parent it is indexing.

        Examples:
            >>> IndexEntry(slice(5, 10)).length(100)
            5
            >>> len(list(range(100))[5:10])
            5
            >>> IndexEntry(slice(5, 100)).length(50)
            45
            >>> len(list(range(50))[5:100])
            45
            >>> IndexEntry(0).length(10)
            1

        Args:
            parent_length (int): The length of the target that this IndexEntry is indexing.

        Returns:
            int: The length of the index if it were applied to a parent of the given length.
        """
        if parent_length == 0:
            return 0
        elif not self.subscriptable():
            return 1
        elif isinstance(self.value, slice):
            return slice_length(self.value, parent_length)
        elif isinstance(self.value, tuple):
            return tuple_length(self.value, parent_length)
        else:
            return 0

    def validate(self, parent_length: int):
        """Checks that the index is not accessing values outside the range of the parent."""

        # Slices are okay, as an out-of-range slice will just yield no samples
        # Check each index of a tuple
        if isinstance(self.value, tuple):
            for idx in self.value:
                IndexEntry(idx).validate(parent_length)

        # Check ints that are too large (positive or negative)
        if isinstance(self.value, int):
            if self.value >= parent_length or self.value < -parent_length:
                raise ValueError(
                    f"Index {self.value} is out of range for tensors with length {parent_length}"
                )

    def intersects(self, low_dim: int, high_dim: int):
        if isinstance(self.value, slice):
            # check if low_dim or high_dim are between start + stop
            low_dim_in_slice = is_value_within_slice(self.value, low_dim)
            high_dim_in_slice = is_value_within_slice(self.value, high_dim)
            return low_dim_in_slice or high_dim_in_slice

        if isinstance(self.value, int):
            return low_dim <= self.value and self.value < high_dim

        raise NotImplementedError

    @property
    def low_bound(self) -> int:
        # TODO: docstring

        if isinstance(self.value, int):
            return self.value
        elif isinstance(self.value, slice):
            return self.value.start

        raise NotImplementedError

    @property
    def high_bound(self) -> int:
        # TODO: docstring

        if isinstance(self.value, int):
            return self.value
        elif isinstance(self.value, slice):
            return self.value.stop

        raise NotImplementedError

    def with_bias(self, amount: int, keep_positive: bool = True) -> "IndexEntry":
        # TODO: docstring

        def _bias(v: int):
            if v is None:
                v = 0
            o = v + amount
            if keep_positive and o < 0:
                o = 0
            return o

        if isinstance(self.value, int):
            return IndexEntry(_bias(self.value))

        if isinstance(self.value, slice):
            s = self.value
            new_slice = slice(
                _bias(s.start) if s.start is not None else None,
                _bias(s.stop) if s.stop is not None else None,
                s.step,
            )

            return IndexEntry(new_slice)

        raise NotImplementedError

    def normalize(self, low_value: int = 0) -> "IndexEntry":
        # TODO: docstring

        if isinstance(self.value, int):
            return IndexEntry(low_value)

        if isinstance(self.value, slice):
            s = self.value

            if is_trivial_slice(s):
                new_slice = slice(low_value, None)
            elif s.start is None:
                new_slice = s
            else:
                if s.stop is None:
                    delta = s.start
                else:
                    delta = abs(s.stop - s.start)
                new_slice = slice(low_value, delta + low_value, s.step)

            return IndexEntry(new_slice)

        raise NotImplementedError

    def clamp_upper(self, max_value: int) -> "IndexEntry":
        if isinstance(self.value, int):
            return IndexEntry(min(self.value, max_value))

        if isinstance(self.value, slice):
            s = self.value

            if is_trivial_slice(s):
                new_slice = slice(None, max_value, s.step)
            elif s.start is None:
                new_slice = slice(None, min(max_value, s.stop), s.step)
            else:
                if s.start < 0 or s.stop < 0:
                    # TODO: negative subslices
                    raise NotImplementedError(
                        "Subslices with negatives is not yet supported!"
                    )

                new_slice = slice(
                    min(s.start, max_value), min(s.stop, max_value), s.step
                )

            return IndexEntry(new_slice)

        raise NotImplementedError

    def clamp_lower(self, min_value: int) -> "IndexEntry":
        if isinstance(self.value, int):
            return IndexEntry(max(self.value, min_value))

        if isinstance(self.value, slice):
            s = self.value

            if is_trivial_slice(s) or s.start is None:
                new_slice = slice(min_value, s.stop, s.step)
            else:
                if s.start < 0 or s.stop < 0:
                    # TODO: negative subslices
                    raise NotImplementedError(
                        "Subslices with negatives is not yet supported!"
                    )

                new_slice = slice(max(s.start, min_value), s.stop, s.step)

            return IndexEntry(new_slice)

        raise NotImplementedError


class Index:
    def __init__(
        self,
        item: Union[IndexValue, "Index", List[IndexEntry]] = slice(None),
    ):
        """Initializes an Index from an IndexValue, another Index, or the values from another Index.

        Represents a list of IndexEntry objects corresponding to indexes into each axis of an ndarray.
        """
        if isinstance(item, Index):
            item = item.values

        if not (isinstance(item, list) and isinstance(item[0], IndexEntry)):
            item = [IndexEntry(item)]

        self.values: List[IndexEntry] = item

    def find_axis(self, offset: int = 0):
        """Returns the index for the nth subscriptable axis in the values of an Index.

        Args:
            offset (int): The number of subscriptable axes to skip before returning.
                Defaults to 0, meaning that the first valid axis is returned.

        Returns:
            int: The index of the found axis, or None if no match is found.
        """
        matches = 0
        for idx, entry in enumerate(self.values):
            if entry.subscriptable():
                if matches == offset:
                    return idx
                else:
                    matches += 1
        return None

    def compose_at(self, item: IndexValue, i: Optional[int] = None):
        """Returns a new Index representing the addition of an IndexValue,
        or the composition with a given axis.

        Examples:
            >>> Index([slice(None), slice(None)]).compose_at(5)
            Index([slice(None), slice(None), 5])

            >>> Index([slice(None), slice(5, 10), slice(None)]).compose_at(3, 1)
            Index([slice(None), 8, slice(None)])

        Args:
            item (IndexValue): The value to append or compose with the Index.
            i (int, optional): The axis to compose with the given item.
                Defaults to None, meaning that the item will be appended instead.

        Returns:
            Index: The result of the addition or composition.
        """
        if i is None or i >= len(self.values):
            return Index(self.values + [IndexEntry(item)])
        else:
            new_values = self.values[:i] + [self.values[i][item]] + self.values[i + 1 :]
            return Index(new_values)

    def __getitem__(
        self, item: Union[int, slice, List[int], Tuple[IndexValue], "Index"]
    ):
        """Returns a new Index representing a subscripting with the given item.
        Modeled after NumPy's advanced integer indexing.

        See: https://numpy.org/doc/stable/reference/arrays.indexing.html

        Examples:
            >>> Index([5, slice(None)])[5]
            Index([5, 5])

            >>> Index([5])[5:6]
            Index([5, slice(5, 6)])

            >>> Index()[0, 1, 2:5, 3]
            Index([0, 1, slice(2, 5), 3])

            >>> Index([slice(5, 6)])[(0, 1, 2:5, 3),]
            Index([(5, 1, slice(2, 5), 3)])

        Args:
            item: The contents of the subscript expression to add to this Index.

        Returns:
            Index: The Index representing the result of the subscript operation.

        Raises:
            TypeError: Given item should be another Index,
                or compatible with NumPy's advanced integer indexing.
        """
        if isinstance(item, int) or isinstance(item, slice):
            ax = self.find_axis()
            return self.compose_at(item, ax)
        elif isinstance(item, tuple):
            new_index = self
            for idx, sub_item in enumerate(item):
                ax = new_index.find_axis(offset=idx)
                new_index = new_index.compose_at(sub_item, ax)
            return new_index
        elif isinstance(item, list):
            return self[(tuple(item),)]  # type: ignore
        elif isinstance(item, Index):
            base = self
            for index in item.values:
                value = index.value
                if isinstance(value, tuple):
                    value = (value,)  # type: ignore
                base = base[value]  # type: ignore
            return base
        else:
            raise TypeError(f"Value {item} is of unrecognized type {type(item)}.")

    def apply(self, samples: List[np.ndarray], include_first_value: bool = False):
        """Applies an Index to a list of ndarray samples with the same number of entries
        as the first entry in the Index.
        """

        if include_first_value:
            index_values = tuple(item.value for item in self.values)
        else:
            index_values = tuple(item.value for item in self.values[1:])

        samples = list(arr[index_values] for arr in samples)
        return samples

    def apply_squeeze(self, samples: List[np.ndarray]):
        """Applies the primary axis of an Index to a list of ndarray samples.
        Will either return the list as given, or return the first sample.
        """
        if self.values[0].subscriptable():
            return samples
        else:
            return samples[0]

    def add_trivials(self, target_dimensionality: int) -> "Index":
        dim_values = self.values

        dims_left = target_dimensionality - len(dim_values)
        if dims_left > 0:
            for _ in range(dims_left):
                dim_values.append(IndexEntry(slice(None)))

    def apply_restricted(
        self,
        sample: np.ndarray,
        bias: Tuple[int, ...],
        upper_bound: Tuple[int, ...] = None,
        normalize: bool = False,
    ) -> np.ndarray:
        # TODO: docstring

        self.add_trivials(len(sample.shape))

        biased_values = []
        for i, value in enumerate(self.values):
            biased_entry = value

            if upper_bound is not None:
                biased_entry = biased_entry.clamp_upper(upper_bound[i])

            biased_entry = biased_entry.with_bias(-bias[i])
            biased_value = biased_entry.value

            if normalize:
                biased_entry = biased_entry.normalize()
                biased_value = biased_entry.value
                if biased_value == 0:
                    continue

            biased_values.append(biased_value)

        return np.squeeze(sample[tuple(biased_values)])

    def is_trivial(self) -> bool:
        """Checks if an Index is equivalent to the trivial slice `[:]`, aka slice(None)."""
        return (len(self.values) == 1) and self.values[0].is_trivial()

    def is_single_dim_effective(self) -> bool:
        """Checks if an Index is only modifying the first dimension.

        Examples:
            array[1] - True
            array[:] - False
            array[1, 1] - False
            array[1, :, 1] - False
            array[:, :, :] - False
            array[:, 1] - False
            array[100:120, :, :, :] - True
            array[0, :, :] - True
        """

        if len(self.values) == 1:
            return True

        for value in self.values[1:]:
            if not value.is_trivial():
                return False

        return True

    @property
    def shape(self) -> Tuple[Optional[int], ...]:
        """Returns the max shape this index can create.
        For trivial slices (ex: array[:]), their shape element is `None`.

        Note:
            If you need this to return only numeric values (not `None` values), you should
            use `shape_if_applied_to` instead.

        Examples:
            >>> a = np.ones((100, 100))
            >>> Index([0, slice(5, 10)]).shape  # equiv: tensor[0, 5:10]
            (1, 5)
            >>>  Index([0, slice(None), 1])  # equiv: tensor[0, :, 1]
            (1, None, 1)
        """

        shape: List[Optional[int]] = []
        for value in self.values:
            if value.is_trivial():
                shape.append(None)
            else:
                l = value.length(9999999999)
                shape.append(l)  # TODO: better way to do this
        return tuple(shape)

    def shape_if_applied_to(
        self, shape: Tuple[int, ...], squeeze: bool = False
    ) -> Tuple[int, ...]:
        # TODO: docstring

        output_shape = np.zeros(len(shape), dtype=int)
        self_shape = self.shape

        for i in range(len(shape)):
            if i >= len(self_shape) or self_shape[i] is None:
                # trivial shape should default to the shape being applied to
                output_shape[i] = shape[i]
            else:
                output_shape[i] = min(self_shape[i], shape[i])  # type: ignore

        output_shape = output_shape.tolist()

        if squeeze:
            output_shape = [x for x in output_shape if x != 1]

        return tuple(output_shape)

    def length(self, parent_length: int):
        """Returns the primary length of an Index given the length of the parent it is indexing.
        See: IndexEntry.length"""
        return self.values[0].length(parent_length)

    def validate(self, parent_length):
        """Checks that the index is not accessing values outside the range of the parent."""
        self.values[0].validate(parent_length)

    def intersects(
        self, low_bound: Tuple[int, ...], high_bound: Tuple[int, ...]
    ) -> bool:
        """Checks if the incoming n-dimensional rectangle is intersected by this index object.
        This is useful for tiling, when trying to determine if a tile should be downloaded or not
        (if it exists on this index)."""

        # check if this index overlaps the incoming n dimensional rectangle
        for in_low_dim, in_high_dim, index_entry in zip(
            low_bound, high_bound, self.values
        ):
            entry_low = index_entry.low_bound
            entry_high = index_entry.high_bound

            if index_entry.is_trivial() or entry_low is None or entry_high is None:
                continue
            if in_high_dim <= entry_low:
                return False
            if in_low_dim > entry_high:
                return False

        return True

    @property
    def low_bound(self) -> Tuple[int, ...]:
        """Get the low-bound of this index.

        Example:
            indexing like: array[0:5, 500:505, :]
            returns: (0, 500, None)
        """

        low = []
        for value in self.values:
            low.append(value.low_bound)

        return tuple(low)

    @property
    def high_bound(self) -> Tuple[int, ...]:
        """Get the high-bound of this index.

        Example:
            indexing like: array[0:5, 500:505, :]
            returns: (5, 505, None)
        """

        high = []
        for value in self.values:
            high.append(value.high_bound)

        return tuple(high)

    def split_subslice(self) -> Tuple["Index", "Index"]:
        """Splits the primary axis index from the sample subslice index.

        Examples:
            array[0:5, 10, 5:10]
                - primary:  [0:5]
                - subslice: [10, 5:10]
            array[0]
                - primary:  [0]
                - subslice: [:]

        Returns:
            Tuple[Index, Index]: value0_index, subslice_index
        """

        value0_index = Index([self.values[0]])
        if len(self.values) > 1:
            subslice_index = Index(self.values[1:])
        else:
            subslice_index = Index()

        return value0_index, subslice_index

    def __str__(self):
        values = [entry.value for entry in self.values]
        return f"Index({values})"

    def __repr__(self):
        return f"Index(values={self.values})"
