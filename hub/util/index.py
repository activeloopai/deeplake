from typing import Union


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


class Index:
    def __init__(self, item: Union[int, slice, "Index"] = None):
        """Create a new Index from an int, slice, or another Index."""
        if item is None:
            item = slice(None)

        if isinstance(item, Index):
            item = item.item

        self.item: Union[int, slice] = item

    def __getitem__(self, item: Union[int, slice, "Index"]):
        """Combine the given `item` and this Index.

        Examples:
            >>> Index()[0:100]
            Index(slice(0, 100, None))

            >>> Index()[100:200][5]
            Index(105)

        Args:
            item: The desired sub-index to be composed with this Index.
                Can be an int, slice, or another Index

        Returns:
            The new Index object.

        Raises:
            TypeError: An integer Index should not be indexed further
        """
        if isinstance(item, Index):
            item = item.item

        if isinstance(self.item, int):
            raise TypeError("Subscripting after 'int' is not allowed.")
        elif isinstance(item, int):
            return Index((self.item.start or 0) + item * (self.item.step or 1))
        elif isinstance(item, slice):
            return Index(merge_slices(self.item, item))

    def __str__(self):
        return str(self.to_slice())

    def to_slice(self):
        """Convert this Index into a slice"""
        if isinstance(self.item, int):
            return slice(self.item, self.item + 1)
        else:
            return self.item
