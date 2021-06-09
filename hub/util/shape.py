import numpy as np

from hub.util.exceptions import InvalidShapeIntervalError
from typing import Sequence


def _contains_negatives(shape: Sequence[int]):
    return any(x < 0 for x in shape)


class Shape:
    def __init__(self, lower: Sequence[int], upper: Sequence[int] = None):
        """
        Shapes in hub are best represented as intervals, this is to support dynamic tensors. Instead of having a single tuple of integers representing shape,
        we use 2 tuples of integers to represent the lower and upper bounds of the representing shape. If lower == upper for all cases, the shape is considered
        "fixed". If lower != upper for any cases, the shape is considered "dynamic".

        Args:
            lower (sequence): Sequence of integers that represent the lower-bound shape.
            upper (sequence): Sequence of integers that represent the upper-bound shape. If None is provided, lower is used as upper (implicitly fixed-shape).

        Raises:
            InvalidShapeIntervalError: If the provided lower/upper bounds are incompatible to represent a shape.
        """

        if upper is None:
            upper = lower

        if len(lower) != len(upper):
            raise InvalidShapeIntervalError("Lengths must match.", lower, upper)

        if _contains_negatives(lower):
            raise InvalidShapeIntervalError(
                "Lower cannot contain negative components.", lower=lower
            )

        if _contains_negatives(upper):
            raise InvalidShapeIntervalError(
                "Upper cannot contain negative components.", upper=upper
            )

        if not all(l <= u for l, u in zip(lower, upper)):
            raise InvalidShapeIntervalError(
                "lower[i] must always be <= upper[i].", lower=lower, upper=upper
            )

        self._lower = tuple(lower)
        self._upper = tuple(upper)

    @property
    def is_dynamic(self) -> bool:
        return self.lower != self.upper

    @property
    def is_fixed(self) -> bool:
        return self.lower == self.upper

    @property
    def lower(self):
        return self._lower

    @property
    def upper(self):
        return self._upper

    def __str__(self):
        intervals = []

        for l, u in zip(self.lower, self.upper):
            if l == u:
                intervals.append(str(l))
            else:
                intervals.append("{}:{}".format(l, u))

        return "({})".format(", ".join(intervals))
