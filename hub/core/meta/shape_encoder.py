from hub.constants import SHAPE_META_FILENAME
from typing import Tuple
from hub.core.storage.provider import StorageProvider
import numpy as np


SHAPE_META_DTYPE = np.uint64


class ShapeEncoder:
    def __init__(self, storage: StorageProvider):
        self.storage = storage
        self.key = SHAPE_META_FILENAME
        self._encoded_shapes = None
        self.load_shapes()

    def load_shapes(self):
        if self.key in self.storage:
            # TODO: read/parse from storage
            raise NotImplementedError()

    def __getitem__(self, sample_index: int) -> np.ndarray:
        if self.num_samples == 0:
            raise IndexError(
                f"Index {sample_index} is out of bounds for an empty shape meta."
            )

        if sample_index < 0:
            sample_index = (self.num_samples) + sample_index

        shape_index = np.searchsorted(self._encoded_shapes[:, -1], sample_index)
        return tuple(self._encoded_shapes[shape_index, :-1])

    @property
    def num_samples(self) -> int:
        if self._encoded_shapes is None:
            return 0
        return int(self._encoded_shapes[-1, -1] + 1)

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
                self._encoded_shapes[-1, -1] += count

            else:
                last_shape_index = self._encoded_shapes[-1, -1]
                shape_entry = np.array(
                    [[*shape, last_shape_index + count]], dtype=SHAPE_META_DTYPE
                )

                self._encoded_shapes = np.concatenate(
                    [self._encoded_shapes, shape_entry], axis=0
                )

        else:
            self._encoded_shapes = np.array(
                [[*shape, count - 1]], dtype=SHAPE_META_DTYPE
            )
