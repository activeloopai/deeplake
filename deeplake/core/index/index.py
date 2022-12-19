from typing import Union, List, Tuple, Iterable, Optional
from collections.abc import Iterable
import numpy as np

IndexValue = Union[int, slice, Tuple[int, ...]]


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

    ``x[existing_slice][new_slice] == x[merge_slices(existing_slice, new_slice)]``

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
    """Returns the ``i`` th element of a slice ``s``.

    Examples:
        >>> slice_at_int(slice(None), 10)
        10

        >>> slice_at_int(slice(10, 20, 2), 3)
        16

    Args:
        s (slice): The slice to index into.
        i (int): The integer offset into the slice.

    Returns:
        int: The index corresponding to the offset into the slice.

    Raises:
        NotImplementedError: Nontrivial slices should not be indexed with negative integers.
        IndexError: If step is negative and start is not greater than stop.
    """
    if s == slice(None):
        return i

    if i < 0:
        raise NotImplementedError(
            "Subscripting slices with negative integers is not supported."
        )

    step = s.step if s.step is not None else 1

    if step < 0:
        if (s.start and s.stop) and (s.stop > s.start):
            raise IndexError(f"index {i} out of bounds.")

    start = s.start

    if start is None:
        start = -1 if step < 0 else 0

    return start + i * step


def slice_length(s: slice, parent_length: int) -> int:
    """Returns the length of a slice given the length of its parent."""
    start, stop, step = s.indices(parent_length)
    step_offset = step - (
        1 if step > 0 else -1
    )  # Used to ceil/floor depending on step direction
    slice_length = stop - start
    total_length = (slice_length + step_offset) // step
    return max(0, total_length)


def replace_ellipsis_with_slices(items, ndim: int):
    if items is Ellipsis:
        return (slice(None),) * ndim
    try:
        idx = items.index(Ellipsis)
    except ValueError:
        return items
    nslices = ndim - len(items) + 1
    if Ellipsis in items[idx + 1 :]:
        raise IndexError("an index can only have a single ellipsis ('...')")
    items = items[:idx] + (slice(None),) * nslices + items[idx + 1 :]
    return items


class IndexEntry:
    def __init__(self, value: IndexValue = slice(None)):
        self.value = value

    def __str__(self):
        return f"IndexEntry({self.value})"

    def __getitem__(self, item: IndexValue):
        """Combines the given ``item`` and this IndexEntry.
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
            IndexEntry: The new IndexEntry object.

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
            elif isinstance(item, (tuple, list)):
                new_value = tuple(slice_at_int(self.value, idx) for idx in item)
                return IndexEntry(new_value)
        elif isinstance(self.value, (tuple, list)):
            if isinstance(item, int) or isinstance(item, slice):
                return IndexEntry(self.value[item])
            elif isinstance(item, (tuple, list)):
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
        elif isinstance(self.value, Iterable):
            yield from map(parse_int, self.value)
        elif callable(self.value):
            yield from self.value()  # type: ignore

    def is_trivial(self):
        """Checks if an IndexEntry represents the entire slice"""
        return (
            isinstance(self.value, slice)
            and not self.value.start
            and self.value.stop is None
            and ((self.value.step or 1) == 1)
        )

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
        lenf = getattr(self.value, "__len__", None)
        if lenf is None:
            return 0
        return lenf()

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
                raise IndexError(
                    f"Index {self.value} is out of range for tensors with length {parent_length}"
                )

    def downsample(self, factor: int, length: int):
        """Downsamples an IndexEntry by a given factor.

        Args:
            factor (int): The factor by which to downsample.
            length (int): The length of the downsampled IndexEntry.

        Returns:
            IndexEntry: The downsampled IndexEntry.

        Raises:
            TypeError: If the IndexEntry cannot be downsampled.
        """
        if isinstance(self.value, slice):
            start = self.value.start or 0
            stop = self.value.stop
            step = self.value.step or 1
            assert step == 1, "Cannot downsample with step != 1"
            downsampled_start = start // factor
            downsampled_stop = stop // factor if stop is not None else None
            if (
                downsampled_stop is None
                or downsampled_stop - downsampled_start != length
            ):
                downsampled_stop = downsampled_start + length
            return IndexEntry(slice(downsampled_start, downsampled_stop, 1))
        else:
            raise TypeError(
                f"Cannot downsample IndexEntry with value {self.value} of type {type(self.value)}"
            )


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
        elif item in ((), [], None):
            item = slice(None)

        if isinstance(item, tuple):
            item = list(map(IndexEntry, item))

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
            return self[tuple(v.value for v in item.values)]  # type: ignore
        else:
            raise TypeError(f"Value {item} is of unrecognized type {type(item)}.")

    def apply(self, samples: List[np.ndarray]):
        """Applies an Index to a list of ndarray samples with the same number of entries
        as the first entry in the Index.
        """
        index_values = tuple(item.value for item in self.values[1:])
        if index_values:
            samples = [arr[index_values] for arr in samples]
        else:
            samples = list(samples)
        return samples

    def apply_squeeze(self, samples: List[np.ndarray]):
        """Applies the primary axis of an Index to a list of ndarray samples.
        Will either return the list as given, or return the first sample.
        """
        if self.values[0].subscriptable():
            return samples
        else:
            return samples[0]

    def is_trivial(self):
        """Checks if an Index is equivalent to the trivial slice `[:]`, aka slice(None)."""
        return (len(self.values) == 1) and self.values[0].is_trivial()

    def length(self, parent_length: int):
        """Returns the primary length of an Index given the length of the parent it is indexing.
        See: :meth:`IndexEntry.length`"""
        return self.values[0].length(parent_length)

    def validate(self, parent_length):
        """Checks that the index is not accessing values outside the range of the parent."""
        self.values[0].validate(parent_length)

    def __str__(self):
        eval_f = lambda v: list(v()) if callable(v) else v
        values = [eval_f(entry.value) for entry in self.values]
        return f"Index({values})"

    def __repr__(self):
        return f"Index(values={self.values})"

    def to_json(self):
        ret = []
        for e in self.values:
            v = e.value
            if isinstance(v, slice):
                ret.append({"start": v.start, "stop": v.stop, "step": v.step})
            elif isinstance(v, Iterable):
                ret.append(list(v))
            elif callable(v):
                ret.append(list(v()))
            else:
                ret.append(v)
        return ret

    @classmethod
    def from_json(cls, idxs):
        entries = []
        for idx in idxs:
            if isinstance(idx, dict):
                idx = slice(idx["start"], idx["stop"], idx["step"])
            entries.append(IndexEntry(idx))
        return cls(entries)

    def __len__(self):
        return len(self.values)

    def subscriptable_at(self, i: int) -> bool:
        try:
            return self.values[i].subscriptable()
        except IndexError:
            return True

    def length_at(self, i: int, parent_length: int) -> int:
        try:
            return self.values[i].length(parent_length)
        except IndexError:
            return parent_length

    def trivial_at(self, i: int) -> bool:
        try:
            return self.values[i].is_trivial()
        except IndexError:
            return True

    def downsample(self, factor: int, shape: Tuple[int, ...]):
        """Downsamples an Index by the given factor.

        Args:
            factor (int): The factor to downsample by.
            shape (Tuple[int, ...]): The shape of the downsampled data.

        Returns:
            Index: The downsampled Index.
        """
        new_values = [
            v.downsample(factor, length) for v, length in zip(self.values[:2], shape)
        ]
        new_values += self.values[2:]
        return Index(new_values)
