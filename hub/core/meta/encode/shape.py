from hub.core.meta.encode.base_encoder import Encoder
from hub.constants import ENCODING_DTYPE
from typing import Tuple
from hub.core.storage.provider import StorageProvider
import numpy as np


class ShapeEncoder(Encoder):
    """Custom compressor that allows reading of shapes from a sample index without decompressing.
    Requires that all shapes encoded have the same dimensionality (`len(shape)`).

    Layout:
        `_encoded` is a 2D array.

        Rows:
            The number of rows is equal to the number of unique runs of shapes that exist upon ingestion. See examples below.

        Columns:
            The number of columns is equal to the dimensionality (`len(shape)`) of the shapes + 1.
            Each row looks like this: [shape_dim0, shape_dim1, shape_dim2, ..., last_index], where `last_index`
            is equal to the last index the specified shape in that row exists. This means that a shape can be shared
            by multiple samples, so long as they were added directly after each other. See examples below.

        Fixed Example:
            >>> enc = ShapeEncoder()
            >>> enc.register_samples((1,), 100)  # represents scalar values
            >>> enc._encoded
            [[1, 99]]
            >>> enc.register_samples((1,), 10000)
            >>> enc._encoded
            [[1, 10099]]
            >>> enc.num_samples
            10100
            >>> enc[5000]
            (1,)

        Dynamic Example:
            >>> enc = ShapeEncoder()
            >>> enc.register_samples((28, 28), 1)
            >>> enc._encoded
            [[28, 28, 0]]
            >>> enc.register_samples((28, 28, 10))
            >>> enc._encoded
            [[28, 28, 10]]
            >>> enc.register_samples((29, 28, 5))
            >>> enc._encoded
            [[28, 28, 10],
             [29, 28, 15]]
            >>> enc.register_samples((28, 28, 3))
            >>> enc._encoded
            [[28, 28, 10],
             [29, 28, 15],
             [28, 28, 18]]
            >>> enc.num_samples
            19
            >>> enc[10]
            (28, 28)
            >>> enc[11]
            (29, 28)

        Best case scenario:
            The best case scenario is when all samples have the same shape. This means that only 1 row is created.
            This is O(1) lookup.

        Worst case scenario:
            The worst case scenario is when all samples have different shapes. This means that there are as many rows as there are samples.
            This is O(log(N)) lookup.

        Lookup algorithm:
            To get the shape for some sample index, you do a binary search over the right-most column. This will give you
            the row that corresponds to that sample index (since the right-most column is our "last index" for that shape).
            Then, you use all elements to the left as your shape!

    Args:
        encoded_shapes (np.ndarray): Encoded shapes that this instance should start with. Defaults to None.
    """

    @property
    def array(self):
        return self._encoded

    def derive_value(self, row: np.ndarray, *_) -> np.ndarray:
        return tuple(row[: self.last_index_index])

    def validate_incoming_item(self, shape: Tuple[int]):
        if len(self._encoded) > 0:
            last_shape = self[-1]  # TODO: optimize this

            if len(shape) != len(last_shape):
                raise ValueError(
                    f"All sample shapes in a tensor must have the same len(shape). Expected: {len(last_shape)} got: {len(shape)}."
                )

    def combine_condition(self, shape: Tuple[int]) -> bool:
        last_shape = self[-1]  # TODO: optimize this

        return shape == last_shape
