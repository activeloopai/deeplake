import numpy as np

from hub.util.exceptions import InvalidShapeIntervalError
from typing import Iterable


def _contains_negatives(shape: Iterable[int]):
    return any([x < 0 for x in shape])


class Shape:
    def __init__(self, lower: Iterable[int], upper: Iterable[int]=None):
        if upper is None:
            upper = lower
        
        if len(lower) != len(upper):
            raise InvalidShapeIntervalError("Lengths must match.", lower, upper)

        if _contains_negatives(lower):
            raise InvalidShapeIntervalError("Lower cannot contain negative components.", lower=lower)

        if _contains_negatives(upper):
            raise InvalidShapeIntervalError("Upper cannot contain negative components.", upper=upper)

        if not (np.asarray(lower) <= np.asarray(upper)).all():
            raise InvalidShapeIntervalError("lower[i] must always be <= upper[i].", lower=lower, upper=upper)

        self._lower = tuple(lower)
        self._upper = tuple(upper)

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