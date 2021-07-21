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
            >>> enc.add_shape((1,), 100)  # represents scalar values
            >>> enc._encoded
            [[1, 99]]
            >>> enc.add_shape((1,), 10000)
            >>> enc._encoded
            [[1, 10099]]
            >>> enc.num_samples
            10100
            >>> enc[5000]
            (1,)

        Dynamic Example:
            >>> enc = ShapeEncoder()
            >>> enc.add_shape((28, 28), 1)
            >>> enc._encoded
            [[28, 28, 0]]
            >>> enc.add_shape((28, 28, 10))
            >>> enc._encoded
            [[28, 28, 10]]
            >>> enc.add_shape((29, 28, 5))
            >>> enc._encoded
            [[28, 28, 10],
             [29, 28, 15]]
            >>> enc.add_shape((28, 28, 3))
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

    def __getitem__(self, sample_index: int) -> np.ndarray:
        if self.num_samples == 0:
            raise IndexError(
                f"Index {sample_index} is out of bounds for an empty shape encoding."
            )

        if sample_index < 0:
            sample_index = (self.num_samples) + sample_index

        idx = np.searchsorted(self._encoded[:, -1], sample_index)
        return tuple(self._encoded[idx, :-1])

    @property
    def nbytes(self):
        return self._encoded.nbytes

    @property
    def array(self):
        return self._encoded

    def add_shape(
        self,
        shape: Tuple[int],
        count: int,
    ):

        if count <= 0:
            raise ValueError(f"Shape `count` should be > 0. Got {count}.")

        if self.num_samples != 0:
            last_shape = self[-1]

            if len(shape) != len(last_shape):
                raise ValueError(
                    f"All sample shapes in a tensor must have the same len(shape). Expected: {len(last_shape)} got: {len(shape)}."
                )

            if shape == last_shape:
                # increment last shape's index by `count`
                self._encoded[-1, self.last_index_index] += count

            else:
                last_shape_index = self._encoded[-1, self.last_index_index]
                shape_entry = np.array(
                    [[*shape, last_shape_index + count]], dtype=ENCODING_DTYPE
                )

                self._encoded = np.concatenate([self._encoded, shape_entry], axis=0)

        else:
            self._encoded = np.array([[*shape, count - 1]], dtype=ENCODING_DTYPE)

    def update_shape(self, local_sample_index: int, new_shape: Tuple[int]):
        # TODO: this function needs optimization

        encoded_index = self.translate_index(local_sample_index)

        entry = self._encoded_byte_positions[encoded_index]

        if self[-1] == new_shape:
            return

        # TODO: if shapes don't match and there is only 1 sample for this row, all you need to do is update shape
        # TODO: if shapes don't match and there is > 1 sample for this row, you will need to create a new row

        raise NotImplementedError
