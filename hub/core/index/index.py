from typing import Union, List, Iterable
from dataclasses import dataclass


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
    """

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
    """
    return (s.start or 0) + i * (s.step or 1)


@dataclass
class IndexItem:
    value: Union[int, slice, Tuple[int]] = slice(None)

    def __getitem__(self, item: Union[int, slice, Tuple[int]]):
        """Combines the given `item` and this IndexItem.
        Returns a new IndexItem representing the composition of the two.

        Examples:
            >>> IndexItem()[0:100]
            IndexItem(slice(0, 100, None))

            >>> IndexItem()[100:200][5]
            IndexItem(105)

            >>> IndexItem()[(0, 1, 2, 3)]
            IndexItem((0, 1, 2, 3))

            >>> IndexItem()[1, 2, 3]
            IndexItem((0, 1, 2, 3))

        Args:
            item: The desired sub-index to be composed with this IndexItem.
                Can be an int, a slice, or a tuple of ints.

        Returns:
            The new IndexItem object.

        Raises:
            TypeError: An integer IndexItem should not be indexed further.
        """

        if isinstance(self.value, int):
            raise TypeError(
                "Subscripting IndexItem after 'int' is not allowed. Use Index instead."
            )
        elif isinstance(self.value, slice):
            if isinstance(item, int):
                new_value = slice_at_int(self.value, item)
                return IndexItem(new_value)
            elif isinstance(item, slice):
                return IndexItem(merge_slices(self.value, item))
            elif isinstance(item, tuple):
                new_value = tuple(slice_at_int(self.value, idx) for idx in item)
                return IndexItem(new_value)
        elif isinstance(self.value, tuple):
            if isinstance(item, int) or isinstance(item, slice):
                return IndexItem(self.value[item])
            elif isinstance(item, tuple):
                new_value = tuple(self.value[idx] for idx in item)
                return IndexItem(new_value)

        raise TypeError(f"Value {item} is of unrecognized type {type(item)}")

    def indices(length: int):
        """Generates the sequence of integer indices for a target of a given length."""
        if isinstance(self.value, int):
            yield self.value
        elif isinstance(self.value, slice):
            start = self.value.start or 0
            stop = min(length, self.value.stop or length)
            step = self.value.step
            yield from range(start, stop, step)
        elif isinstance(self.value, tuple):
            for i in self.value:
                if i >= length:
                    break
                yield i
