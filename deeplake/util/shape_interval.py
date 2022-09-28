from typing import Sequence

from deeplake.util.exceptions import InvalidShapeIntervalError
from typing import Optional, Sequence, Tuple


def _contains_negatives(shape: Sequence[int]):
    return any(x and x < 0 for x in shape)


class ShapeInterval:
    def __init__(self, lower: Sequence[int], upper: Optional[Sequence[int]] = None):
        """
        Shapes in Deep Lake are best represented as intervals, this is to support dynamic tensors. Instead of having a single tuple of integers representing shape,
        we use 2 tuples of integers to represent the lower and upper bounds of the representing shape.

        - If ``lower == upper`` for all cases, the shape is considered "fixed".
        - If ``lower != upper`` for any cases, the shape is considered "dynamic".

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

        if not all(l is None or u is None or l <= u for l, u in zip(lower, upper)):
            raise InvalidShapeIntervalError(
                "lower[i] must always be <= upper[i].", lower=lower, upper=upper
            )

        self._lower = tuple(lower)
        self._upper = tuple(upper)

    def astuple(self) -> Tuple[Optional[int], ...]:
        # TODO: named tuple? NHWC shape would be (10, 224, 224, 3) could be (N=10, H=224, W=224, C=3).

        shape = []
        for low, up in zip(self.lower, self.upper):
            shape.append(None if low != up else low)  # type: ignore
        return tuple(shape)

    @property
    def is_dynamic(self) -> bool:
        return self.lower != self.upper

    @property
    def lower(self) -> Tuple[int, ...]:
        return self._lower

    @property
    def upper(self) -> Tuple[int, ...]:
        return self._upper

    def __str__(self):
        intervals = []

        for l, u in zip(self.lower, self.upper):
            if l == u:
                intervals.append(str(l))
            else:
                intervals.append(f"{l}:{u}")

        return f"({', '.join(intervals)})"

    def __repr__(self):
        return str(self)
